#  coding=utf-8
#  Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from logging import getLogger
from typing import List, Mapping, Tuple

import numpy as np
from tensorrt_llm import BuilderConfig, Module
from tensorrt_llm import Mapping as ShardingConfig
from tensorrt_llm.functional import LayerNormPositionType, LayerNormType
from tensorrt_llm.models import DecoderModel, WhisperEncoder
from tensorrt_llm.quantization import QuantMode

from optimum.nvidia.configs import ModelConfig, QuantizationConfig
from optimum.nvidia.lang import DataType
from optimum.nvidia.models import ConvertibleModel
from optimum.nvidia.weights import SupportsNpz, SupportsSafetensors, WeightAdapter, as_numpy, retrieve_qkv, shard

from ..runtime import TensorRTForSpeechSeq2Seq


LOGGER = getLogger(__name__)


# TODO: Cleanup the config.config["xxx"] in the future, using Optimum's NormalizedConfig.
class WhisperEncoderWeightAdapter(WeightAdapter, SupportsSafetensors, SupportsNpz):
    """ """

    QUANTIZATION_EXCLUDED_PARAMETERS = {"lm_head"}
    TENSORRT_LLM_MODEL_CLASS = WhisperEncoder
    LAYERS_PREFIX = "model.encoder.layers"

    def convert(
        self,
        model: Module,
        config: ModelConfig,
        builder: BuilderConfig,
        qconfig: QuantizationConfig,
        rank: int,
        weights: Mapping[str, np.array],
    ) -> Module:
        shard_info = self._sharding_config
        precision = DataType(builder.precision)

        # TensorRT-LLM model definition uses a single GEMM for query/key/value, while transformers does not.
        qkv_packed_layers = retrieve_qkv(
            num_layers=config.num_layers,
            layer_prefix=self.LAYERS_PREFIX,
            attn_layer_name="self_attn",
            weights=weights,
            precision=precision,
            use_multi_head_attention=True,  # Whisper never uses GQA/MQA.
            num_kv_heads=config.config["encoder_attention_heads"],
            shard_info=shard_info,
        )

        layers_per_stage = config.num_layers // shard_info.pp_size
        layers_range = range(shard_info.pp_rank * layers_per_stage, (shard_info.pp_rank + 1) * layers_per_stage, 1)

        LOGGER.debug(f"Converting Whisper with dtype: {precision} for rank {rank} and layers: {layers_range}")

        # Convert specific tensors
        if shard_info.is_first_pp_rank():
            # conv1
            conv1_weight = as_numpy(weights["model.encoder.conv1.weight"], precision)
            conv1_bias = as_numpy(weights["model.encoder.conv1.bias"], precision)

            # TensorRT-LLM Conv1d weight is 4D while transformers checkpoint ones are 3D.
            model.conv1.weight.value = conv1_weight[..., None]
            model.conv1.bias.value = conv1_bias

            # conv2
            conv2_weight = as_numpy(weights["model.encoder.conv2.weight"], precision)
            conv2_bias = as_numpy(weights["model.encoder.conv2.bias"], precision)

            model.conv2.weight.value = conv2_weight[..., None]
            model.conv2.bias.value = conv2_bias

            # embed_positions
            position_embeddings = as_numpy(weights["model.encoder.embed_positions.weight"], precision)

            model.positional_embedding.value = position_embeddings

        if shard_info.is_last_pp_rank():
            # Final layer norm
            final_norm_weight = as_numpy(weights["model.encoder.layer_norm.weight"], precision)
            final_norm_bias = as_numpy(weights["model.encoder.layer_norm.bias"], precision)

            model.ln_post.weight.value = final_norm_weight
            model.ln_post.bias.value = final_norm_bias

        # Map all the hidden layers
        for layer_idx in layers_range:
            idx = layer_idx - shard_info.pp_rank * layers_per_stage
            prefix = f"{self.LAYERS_PREFIX}.{idx}"

            # attention_layernorm
            attn_layer_norm_weight = as_numpy(weights[f"{prefix}.self_attn_layer_norm.weight"], precision)
            attn_layer_norm_bias = as_numpy(weights[f"{prefix}.self_attn_layer_norm.bias"], precision)
            model.encoder_layers[idx].attention_layernorm.weight.value = attn_layer_norm_weight
            model.encoder_layers[idx].attention_layernorm.bias.value = attn_layer_norm_bias

            # mlp_layernorm
            mlp_layer_norm_weight = as_numpy(weights[f"{prefix}.final_layer_norm.weight"], precision)
            mlp_layer_norm_bias = as_numpy(weights[f"{prefix}.final_layer_norm.bias"], precision)
            model.encoder_layers[idx].mlp_layernorm.weight.value = mlp_layer_norm_weight
            model.encoder_layers[idx].mlp_layernorm.bias.value = mlp_layer_norm_bias

            # Self attention layer
            qkv_weights, qkv_bias = qkv_packed_layers[idx]
            q_weight, k_weight, v_weight = qkv_weights

            # Shard, Whisper never uses GQA/MQA.
            qkv_weight = np.stack((q_weight, k_weight, v_weight), axis=0)
            rank_tensor = shard(qkv_weight, rank, shard_info.tp_size, axis=1)
            qkv_weight = rank_tensor.reshape(-1, config.hidden_size)

            model.encoder_layers[idx].attention.qkv.weight.value = np.ascontiguousarray(qkv_weight)

            if qkv_bias is not None:
                q_bias, k_bias, v_bias = qkv_bias
                packed_qkv_bias = np.concatenate((q_bias, k_bias, v_bias), axis=0)
                model.encoder_layers[idx].attention.qkv.bias.value = np.ascontiguousarray(packed_qkv_bias)

            # Common projection logic
            for src, dst, shard_axis in [
                (
                    "self_attn.out_proj.weight",
                    model.encoder_layers[idx].attention.dense.weight,
                    1,
                ),  # row tensor parallel.
                ("fc1.weight", model.encoder_layers[idx].mlp.fc.weight, 0),  # column tensor parallel.
                ("fc2.weight", model.encoder_layers[idx].mlp.proj.weight, 1),  # row tensor parallel.
            ]:
                tensor = as_numpy(weights[f"{prefix}.{src}"], precision)
                rank_tensor = shard(tensor, rank, shard_info.tp_size, shard_axis)
                dst.value = np.ascontiguousarray(rank_tensor)

            # Bias is never sharded.
            for src, dst in [
                ("self_attn.out_proj.bias", model.encoder_layers[idx].attention.dense.bias),
                ("fc1.bias", model.encoder_layers[idx].mlp.fc.bias),
                ("fc2.bias", model.encoder_layers[idx].mlp.proj.bias),
            ]:
                tensor = as_numpy(weights[f"{prefix}.{src}"], precision)
                dst.value = tensor

        return model

    @staticmethod
    def allocate_model(
        config: ModelConfig, sharding: ShardingConfig, dtype: DataType, quant_mode: QuantMode
    ) -> Tuple[Module, Module]:
        LOGGER.debug("Allocating WhisperEncoder model...")
        return WhisperEncoderWeightAdapter.TENSORRT_LLM_MODEL_CLASS(
            n_mels=config.config["num_mel_bins"],
            n_ctx=config.config["max_source_positions"],
            n_state=config.hidden_size,
            n_head=config.config["encoder_attention_heads"],
            n_layer=config.config["encoder_layers"],
            dtype=dtype.as_trt(),
        )

    @staticmethod
    def get_scaling_factors(
        weights: Mapping[str, np.array], num_layers: int, mode: QuantMode
    ) -> Mapping[str, List[np.array]]:
        raise NotImplementedError("get_scaling_factors is not implemented for WhisperEncoderWeightAdapter.")


class WhisperDecoderWeightAdapter(WeightAdapter, SupportsSafetensors, SupportsNpz):
    """ """

    QUANTIZATION_EXCLUDED_PARAMETERS = {"lm_head"}
    TENSORRT_LLM_MODEL_CLASS = DecoderModel
    LAYERS_PREFIX = "model.decoder.layers"

    def convert(
        self,
        model: Module,
        config: ModelConfig,
        builder: BuilderConfig,
        qconfig: QuantizationConfig,
        rank: int,
        weights: Mapping[str, np.array],
    ) -> Module:
        shard_info = self._sharding_config
        precision = DataType(builder.precision)

        self_attn_qkv_packed_layers = retrieve_qkv(
            num_layers=config.num_layers,
            layer_prefix=self.LAYERS_PREFIX,
            attn_layer_name="self_attn",
            weights=weights,
            precision=precision,
            use_multi_head_attention=True,  # Whisper never uses MQA/GQA.
            num_kv_heads=config.config["decoder_attention_heads"],
            shard_info=shard_info,
        )

        cross_attn_qkv_packed_layers = retrieve_qkv(
            num_layers=config.num_layers,
            layer_prefix=self.LAYERS_PREFIX,
            attn_layer_name="encoder_attn",
            weights=weights,
            precision=precision,
            use_multi_head_attention=True,  # Whisper never uses MQA/GQA.
            num_kv_heads=config.config["decoder_attention_heads"],
            shard_info=shard_info,
        )

        layers_per_stage = config.num_layers // shard_info.pp_size
        layers_range = range(shard_info.pp_rank * layers_per_stage, (shard_info.pp_rank + 1) * layers_per_stage, 1)

        LOGGER.debug(f"Converting Whisper with dtype: {precision} for rank {rank} and layers: {layers_range}")

        # Convert specific tensors
        if shard_info.is_first_pp_rank():
            # embed_tokens
            tokens_embeddings = as_numpy(weights["model.decoder.embed_tokens.weight"], precision)
            model.embedding.vocab_embedding.weight.value = tokens_embeddings

            # embed_positions
            positions_embeddings = as_numpy(weights["model.decoder.embed_positions.weight"], precision)
            model.embedding.position_embedding.weight.value = positions_embeddings

        if shard_info.is_last_pp_rank():
            # Final layer norm
            final_norm_weight = as_numpy(weights["model.decoder.layer_norm.weight"], precision)
            final_norm_bias = as_numpy(weights["model.decoder.layer_norm.bias"], precision)

            model.final_layernorm.weight.value = final_norm_weight
            model.final_layernorm.bias.value = final_norm_bias

            # Final vocab projection
            lm_head = as_numpy(weights["model.decoder.embed_tokens.weight"], precision)
            rank_tensor = shard(lm_head, rank, shard_info.tp_size)
            model.lm_head.weight.value = rank_tensor

        # Map all the hidden layers
        for layer_idx in layers_range:
            idx = layer_idx - shard_info.pp_rank * layers_per_stage
            prefix = f"{self.LAYERS_PREFIX}.{idx}"

            # self_attention_layernorm
            self_attn_layer_norm_weight = as_numpy(weights[f"{prefix}.self_attn_layer_norm.weight"], precision)
            self_attn_layer_norm_bias = as_numpy(weights[f"{prefix}.self_attn_layer_norm.bias"], precision)

            model.decoder_layers[idx].self_attention_layernorm.weight.value = self_attn_layer_norm_weight
            model.decoder_layers[idx].self_attention_layernorm.bias.value = self_attn_layer_norm_bias

            # cross_attention_layernorm
            encoder_attn_layer_norm_weight = as_numpy(weights[f"{prefix}.encoder_attn_layer_norm.weight"], precision)
            encoder_attn_layer_norm_bias = as_numpy(weights[f"{prefix}.encoder_attn_layer_norm.bias"], precision)
            model.decoder_layers[idx].cross_attention_layernorm.weight.value = encoder_attn_layer_norm_weight
            model.decoder_layers[idx].cross_attention_layernorm.bias.value = encoder_attn_layer_norm_bias

            # mlp_layernorm
            final_layer_norm_weight = as_numpy(weights[f"{prefix}.final_layer_norm.weight"], precision)
            final_layer_norm_bias = as_numpy(weights[f"{prefix}.final_layer_norm.bias"], precision)
            model.decoder_layers[idx].mlp_layernorm.weight.value = final_layer_norm_weight
            model.decoder_layers[idx].mlp_layernorm.bias.value = final_layer_norm_bias

            # Self attention layer
            qkv_weight, qkv_bias = self_attn_qkv_packed_layers[idx]
            q_weight, k_weight, v_weight = qkv_weight

            # Shard - Whisper never uses GQA/MQA.
            qkv_weight = np.stack((q_weight, k_weight, v_weight), axis=0)
            rank_tensor = shard(qkv_weight, rank, shard_info.tp_size, axis=1)
            qkv_weight = rank_tensor.reshape(-1, config.hidden_size)

            model.decoder_layers[idx].self_attention.qkv.weight.value = np.ascontiguousarray(qkv_weight)

            if qkv_bias is not None:
                q_bias, k_bias, v_bias = qkv_bias
                packed_qkv_bias = np.concatenate((q_bias, k_bias, v_bias), axis=0)
                model.decoder_layers[idx].self_attention.qkv.bias.value = np.ascontiguousarray(packed_qkv_bias)

            # Cross attention layer
            qkv_weight, qkv_bias = cross_attn_qkv_packed_layers[idx]
            q_weight, k_weight, v_weight = qkv_weight

            # Shard - Whisper never uses GQA/MQA.
            qkv_weight = np.stack((q_weight, k_weight, v_weight), axis=0)
            rank_tensor = shard(qkv_weight, rank, shard_info.tp_size, axis=1)
            qkv_weight = rank_tensor.reshape(-1, config.hidden_size)

            model.decoder_layers[idx].cross_attention.qkv.weight.value = np.ascontiguousarray(qkv_weight)

            if qkv_bias is not None:
                q_bias, k_bias, v_bias = qkv_bias
                packed_qkv_bias = np.concatenate((q_bias, k_bias, v_bias), axis=0)
                model.decoder_layers[idx].cross_attention.qkv.bias.value = np.ascontiguousarray(packed_qkv_bias)

            # Common projection logic.
            # 0: column tensor parallel, 1: row tensor parallel.
            for src, dst, shard_axis in [
                ("self_attn.out_proj.weight", model.decoder_layers[idx].self_attention.dense.weight, 1),
                ("encoder_attn.out_proj.weight", model.decoder_layers[idx].cross_attention.dense.weight, 1),
                ("fc1.weight", model.decoder_layers[idx].mlp.fc.weight, 0),
                ("fc2.weight", model.decoder_layers[idx].mlp.proj.weight, 1),
            ]:
                tensor = as_numpy(weights[f"{prefix}.{src}"], precision)
                rank_tensor = shard(tensor, rank, shard_info.tp_size, shard_axis)
                dst.value = np.ascontiguousarray(rank_tensor)

            # Bias is never sharded.
            for src, dst in [
                ("self_attn.out_proj.bias", model.decoder_layers[idx].self_attention.dense.bias),
                ("encoder_attn.out_proj.bias", model.decoder_layers[idx].cross_attention.dense.bias),
                ("fc1.bias", model.decoder_layers[idx].mlp.fc.bias),
                ("fc2.bias", model.decoder_layers[idx].mlp.proj.bias),
            ]:
                tensor = as_numpy(weights[f"{prefix}.{src}"], precision)
                dst.value = tensor

        return model

    @staticmethod
    def allocate_model(
        config: ModelConfig, sharding: ShardingConfig, dtype: DataType, quant_mode: QuantMode
    ) -> Tuple[Module, Module]:
        LOGGER.debug("Allocating DecoderModel model...")

        # DecoderModel has no quant_mode.
        return DecoderModel(
            num_layers=config.config["decoder_layers"],
            num_heads=config.config["decoder_attention_heads"],
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.config["decoder_ffn_dim"],
            encoder_num_heads=config.config["encoder_attention_heads"],
            encoder_hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            dtype=dtype.as_trt(),
            logits_dtype=DataType.FLOAT32.as_trt(),
            max_position_embeddings=config.config["max_target_positions"],
            has_position_embedding=True,
            relative_attention=False,
            head_size=None,
            encoder_head_size=None,
            num_kv_heads=None,
            encoder_num_kv_heads=None,
            type_vocab_size=None,
            max_distance=0,
            num_buckets=0,
            has_embedding_layernorm=False,
            has_embedding_scale=False,
            q_scaling=1.0,
            has_attention_qkvo_bias=True,
            has_mlp_bias=True,
            has_model_final_layernorm=True,
            layernorm_eps=1e-5,
            layernorm_position=LayerNormPositionType.pre_layernorm,
            layernorm_type=LayerNormType.LayerNorm,
            hidden_act="gelu",
            rescale_before_lm_head=False,
            mapping=sharding,
        )

    @staticmethod
    def get_scaling_factors(
        weights: Mapping[str, np.array], num_layers: int, mode: QuantMode
    ) -> Mapping[str, List[np.array]]:
        raise NotImplementedError("get_scaling_factors is not implemented for WhisperDecoderWeightAdapter.")


class OptimumWhisperEncoder(ConvertibleModel, TensorRTForSpeechSeq2Seq):
    ADAPTER = WhisperEncoderWeightAdapter
    TARGET = WhisperEncoder


class OptimumWhisperDecoder(ConvertibleModel, TensorRTForSpeechSeq2Seq):
    ADAPTER = WhisperDecoderWeightAdapter
    TARGET = DecoderModel
