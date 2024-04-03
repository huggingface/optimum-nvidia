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
import json
from typing import TYPE_CHECKING
from tensorrt_llm.runtime.session import Session
from tensorrt_llm.runtime import GenerationSession, ModelConfig
from tensorrt_llm.builder import BuildConfig

from transformers import GenerationConfig

if TYPE_CHECKING:
    from transformers import PretrainedConfig

import pathlib
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tensorrt_llm import Mapping, str_dtype_to_torch, mpi_rank
from tensorrt_llm._utils import str_dtype_to_trt, torch_to_numpy
from tensorrt_llm.functional import LayerNormPositionType, LayerNormType
from tensorrt_llm.models import DecoderModel as TrtDecoderModel
from tensorrt_llm.models import PretrainedConfig, PretrainedModel
from tensorrt_llm.models import WhisperEncoder as TrtWhisperEncoder
from tensorrt_llm.plugin import PluginConfig
from transformers import PretrainedConfig as TransformersPretrainedConfig
from transformers import PreTrainedModel as TransformersPretrainedModel
from transformers.models.whisper.modeling_whisper import (
    WhisperDecoder as TransformersWhisperDecoder,
)
from transformers.models.whisper.modeling_whisper import (
    WhisperEncoder as TransformersWhisperEncoder,
)
from transformers.models.whisper.modeling_whisper import (
    WhisperForConditionalGeneration as TransformersWhisperForConditionalGeneration,
)

from optimum.nvidia import TensorRTConfig
from optimum.nvidia.config import dtype_to_str
from optimum.nvidia.hub import HuggingFaceHubModel
from optimum.nvidia.runtime import TensorRTForSpeechSeq2Seq
from optimum.nvidia.utils.nvml import get_max_memory


LOGGER = getLogger(__name__)


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    return np.split(v, tp_size, axis=dim)[idx]


def get_qkv(model_params, torch_dtype, attn_prefix: str):
    q_weight = torch_to_numpy(
        model_params[f"{attn_prefix}.q_proj.weight"].to(torch_dtype)
    )
    k_weight = torch_to_numpy(
        model_params[f"{attn_prefix}.k_proj.weight"].to(torch_dtype)
    )
    v_weight = torch_to_numpy(
        model_params[f"{attn_prefix}.v_proj.weight"].to(torch_dtype)
    )

    qkv_weight = (q_weight, k_weight, v_weight)

    # At least one of the query, key, value projection has bias.
    if any(
        bias_name in model_params
        for bias_name in [
            f"{attn_prefix}.q_proj.bias",
            f"{attn_prefix}.k_proj.bias",
            f"{attn_prefix}.v_proj.bias",
        ]
    ):
        # For example whisper encoder k_proj does not have bias, so we will just fill with zeros the fused bias if needed.
        numpy_precision = q_weight.dtype
        if f"{attn_prefix}.q_proj.bias" in model_params:
            q_bias = torch_to_numpy(
                model_params[f"{attn_prefix}.q_proj.bias"].to(torch_dtype)
            )
        else:
            q_bias = np.zeros(q_weight.shape[0], dtype=numpy_precision)

        if f"{attn_prefix}.k_proj.bias" in model_params:
            k_bias = torch_to_numpy(
                model_params[f"{attn_prefix}.k_proj.bias"].to(torch_dtype)
            )
        else:
            k_bias = np.zeros(k_weight.shape[0], dtype=numpy_precision)

        if f"{attn_prefix}.v_proj.bias" in model_params:
            v_bias = torch_to_numpy(
                model_params[f"{attn_prefix}.v_proj.bias"].to(torch_dtype)
            )
        else:
            v_bias = np.zeros(v_weight.shape[0], dtype=numpy_precision)
        qkv_bias = (q_bias, k_bias, v_bias)
    else:
        qkv_bias = None

    return qkv_weight, qkv_bias


def convert_from_hf_whisper_encoder(
    hf_whisper_encoder,
    mapping=Mapping(),
    dtype="float16",
):
    num_layers = hf_whisper_encoder.config.encoder_layers
    torch_dtype = str_dtype_to_torch(dtype)

    model_params = dict(hf_whisper_encoder.named_parameters())
    weights = {}

    # Convert specific tensors
    # if mapping.is_first_pp_rank():
    # conv1
    # TensorRT-LLM Conv1d weight is 4D while transformers checkpoint ones are 3D.
    conv1_weight = torch_to_numpy(model_params["conv1.weight"].to(torch_dtype))
    conv1_weight = conv1_weight[..., None]

    weights["model.conv1.weight"] = conv1_weight
    weights["model.conv1.bias"] = torch_to_numpy(
        model_params["conv1.bias"].to(torch_dtype)
    )

    # conv2
    conv2_weight = torch_to_numpy(model_params["conv2.weight"].to(torch_dtype))
    conv2_weight = conv2_weight[..., None]

    weights["model.conv2.weight"] = conv2_weight
    weights["model.conv2.bias"] = torch_to_numpy(
        model_params["conv2.bias"].to(torch_dtype)
    )

    # embed_positions
    # NOTE: this one is kept as fp32 in Whisper, is this important?
    weights["model.positional_embedding"] = torch_to_numpy(
        model_params["embed_positions.weight"] #.to(torch_dtype)
    )

    # if mapping.is_last_pp_rank():
    # Final layer norm
    weights["model.ln_post.weight"] = torch_to_numpy(
        model_params["layer_norm.weight"].to(torch_dtype)
    )
    weights["model.ln_post.bias"] = torch_to_numpy(
        model_params["layer_norm.bias"].to(torch_dtype)
    )

    # Map all the hidden layers
    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}"

        # attention_layernorm
        weights[f"model.encoder_layers.{layer_idx}.attention_layernorm.weight"] = (
            torch_to_numpy(
                model_params[f"{prefix}.self_attn_layer_norm.weight"].to(torch_dtype)
            )
        )
        weights[f"model.encoder_layers.{layer_idx}.attention_layernorm.bias"] = (
            torch_to_numpy(
                model_params[f"{prefix}.self_attn_layer_norm.bias"].to(torch_dtype)
            )
        )

        # mlp_layernorm
        weights[f"model.encoder_layers.{layer_idx}.mlp_layernorm.weight"] = (
            torch_to_numpy(
                model_params[f"{prefix}.final_layer_norm.weight"].to(torch_dtype)
            )
        )
        weights[f"model.encoder_layers.{layer_idx}.mlp_layernorm.bias"] = (
            torch_to_numpy(
                model_params[f"{prefix}.final_layer_norm.bias"].to(torch_dtype)
            )
        )

        # Self attention layer
        # TensorRT-LLM model definition uses a single GEMM for query/key/value, while transformers does not.
        qkv_weight, qkv_bias = get_qkv(
            model_params, attn_prefix=f"{prefix}.self_attn", torch_dtype=torch_dtype
        )
        q_weight, k_weight, v_weight = qkv_weight

        qkv_weight = np.concatenate((q_weight, k_weight, v_weight), axis=0)
        weight = split(qkv_weight, mapping.tp_size, mapping.tp_rank, dim=1)
        weights[f"model.encoder_layers.{layer_idx}.attention.qkv.weight"] = weight

        if qkv_bias is not None:
            q_bias, k_bias, v_bias = qkv_bias
            packed_qkv_bias = np.concatenate((q_bias, k_bias, v_bias), axis=0)
            weights[f"model.encoder_layers.{layer_idx}.attention.qkv.bias"] = (
                np.ascontiguousarray(packed_qkv_bias)
            )

        # Common projection logic
        # 0: column tensor parallel, 1: row tensor parallel.
        for src, dst, shard_axis in [
            ("self_attn.out_proj.weight", "attention.dense.weight", 1),
            ("fc1.weight", "mlp.fc.weight", 0),
            ("fc2.weight", "mlp.proj.weight", 1),
        ]:
            weight = torch_to_numpy(model_params[f"{prefix}.{src}"].to(torch_dtype))
            weight = split(weight, mapping.tp_size, mapping.tp_rank, dim=shard_axis)
            weights[f"model.encoder_layers.{layer_idx}.{dst}"] = weight

        # Bias is never sharded.
        for src, dst in [
            ("self_attn.out_proj.bias", "attention.dense.bias"),
            ("fc1.bias", "mlp.fc.bias"),
            ("fc2.bias", "mlp.proj.bias"),
        ]:
            weights[f"model.encoder_layers.{layer_idx}.{dst}"] = torch_to_numpy(
                model_params[f"{prefix}.{src}"].to(torch_dtype)
            )

    # weights["lm_head.weight"] = np.zeros((0))  # Just a hack for commands/build.py

    return weights


def convert_from_hf_whisper_decoder(
    hf_whisper_decoder,
    mapping=Mapping(),
    dtype="float16",
):
    weights = {}

    num_layers = hf_whisper_decoder.config.decoder_layers
    torch_dtype = str_dtype_to_torch(dtype)

    model_params = dict(hf_whisper_decoder.named_parameters())

    # Convert specific tensors
    if mapping.is_first_pp_rank():
        # embed_tokens
        weights["model.embedding.vocab_embedding.weight"] = torch_to_numpy(
            model_params["embed_tokens.weight"].to(torch_dtype)
        )

        # embed_positions
        weights["model.embedding.position_embedding.weight"] = torch_to_numpy(
            model_params["embed_positions.weight"].to(torch_dtype)
        )

    if mapping.is_last_pp_rank():
        # Final layer norm
        weights["model.final_layernorm.weight"] = torch_to_numpy(
            model_params["layer_norm.weight"].to(torch_dtype)
        )
        weights["model.final_layernorm.bias"] = torch_to_numpy(
            model_params["layer_norm.bias"].to(torch_dtype)
        )

        # Final vocab projection
        lm_head = torch_to_numpy(model_params["embed_tokens.weight"].to(torch_dtype))
        lm_head = split(lm_head, mapping.tp_size, mapping.tp_rank)
        weights["model.lm_head.weight"] = lm_head

    # Map all the hidden layers
    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}"
        trt_llm_prefix = f"model.decoder_layers.{layer_idx}"

        # self_attention_layernorm
        weights[f"{trt_llm_prefix}.self_attention_layernorm.weight"] = torch_to_numpy(
            model_params[f"{prefix}.self_attn_layer_norm.weight"].to(torch_dtype)
        )
        weights[f"{trt_llm_prefix}.self_attention_layernorm.bias"] = torch_to_numpy(
            model_params[f"{prefix}.self_attn_layer_norm.bias"].to(torch_dtype)
        )

        # cross_attention_layernorm
        weights[f"{trt_llm_prefix}.cross_attention_layernorm.weight"] = torch_to_numpy(
            model_params[f"{prefix}.encoder_attn_layer_norm.weight"].to(torch_dtype)
        )
        weights[f"{trt_llm_prefix}.cross_attention_layernorm.bias"] = torch_to_numpy(
            model_params[f"{prefix}.encoder_attn_layer_norm.bias"].to(torch_dtype)
        )

        # mlp_layernorm
        weights[f"{trt_llm_prefix}.mlp_layernorm.weight"] = torch_to_numpy(
            model_params[f"{prefix}.final_layer_norm.weight"].to(torch_dtype)
        )
        weights[f"{trt_llm_prefix}.mlp_layernorm.bias"] = torch_to_numpy(
            model_params[f"{prefix}.final_layer_norm.bias"].to(torch_dtype)
        )

        # Self attention layer
        qkv_weight, qkv_bias = get_qkv(
            model_params, attn_prefix=f"{prefix}.self_attn", torch_dtype=torch_dtype
        )
        q_weight, k_weight, v_weight = qkv_weight

        qkv_weight = np.concatenate((q_weight, k_weight, v_weight), axis=0)
        weight = split(qkv_weight, mapping.tp_size, mapping.tp_rank, dim=1)
        weights[f"{trt_llm_prefix}.self_attention.qkv.weight"] = weight

        if qkv_bias is not None:
            q_bias, k_bias, v_bias = qkv_bias
            packed_qkv_bias = np.concatenate((q_bias, k_bias, v_bias), axis=0)
            weights[f"{trt_llm_prefix}.self_attention.qkv.bias"] = np.ascontiguousarray(
                packed_qkv_bias
            )

        # Cross attention layer
        qkv_weight, qkv_bias = get_qkv(
            model_params, attn_prefix=f"{prefix}.encoder_attn", torch_dtype=torch_dtype
        )
        q_weight, k_weight, v_weight = qkv_weight

        qkv_weight = np.concatenate((q_weight, k_weight, v_weight), axis=0)
        weight = split(qkv_weight, mapping.tp_size, mapping.tp_rank, dim=1)
        weights[f"{trt_llm_prefix}.cross_attention.qkv.weight"] = weight

        if qkv_bias is not None:
            q_bias, k_bias, v_bias = qkv_bias
            packed_qkv_bias = np.concatenate((q_bias, k_bias, v_bias), axis=0)
            weights[f"{trt_llm_prefix}.cross_attention.qkv.bias"] = (
                np.ascontiguousarray(packed_qkv_bias)
            )

        # Common projection logic.
        # 0: column tensor parallel, 1: row tensor parallel.
        for src, dst, shard_axis in [
            ("self_attn.out_proj.weight", "self_attention.dense.weight", 1),
            ("encoder_attn.out_proj.weight", "cross_attention.dense.weight", 1),
            ("fc1.weight", "mlp.fc.weight", 0),
            ("fc2.weight", "mlp.proj.weight", 1),
        ]:
            weight = torch_to_numpy(model_params[f"{prefix}.{src}"].to(torch_dtype))
            weight = split(weight, mapping.tp_size, mapping.tp_rank, dim=shard_axis)
            weights[f"{trt_llm_prefix}.{dst}"] = weight

        # Bias is never sharded.
        for src, dst in [
            ("self_attn.out_proj.bias", "self_attention.dense.bias"),
            ("encoder_attn.out_proj.bias", "cross_attention.dense.bias"),
            ("fc1.bias", "mlp.fc.bias"),
            ("fc2.bias", "mlp.proj.bias"),
        ]:
            weights[f"{trt_llm_prefix}.{dst}"] = torch_to_numpy(
                model_params[f"{prefix}.{src}"].to(torch_dtype)
            )

    return weights


class WhisperEncoderConfig(TensorRTConfig):
    @classmethod
    def from_config(cls, config: TransformersPretrainedConfig) -> "TensorRTConfig":
        # Retrieve the quantization from the transformers config (if provided)
        _, qconfig = TensorRTConfig.get_quantization_config(config)

        trt_config = cls(
            architecture=config.architectures[0],
            dtype=dtype_to_str(config.torch_dtype),  # TODO: always float32?
            logits_dtype="float32",
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_target_positions,
            hidden_size=config.d_model,
            num_hidden_layers=config.encoder_layers,
            num_attention_heads=config.encoder_attention_heads,
            num_key_value_heads=config.encoder_attention_heads,
            hidden_act=config.activation_function,
            intermediate_size=None,
            norm_epsilon=None,
            position_embedding_type="learned_absolute",
            world_size=1,
            tp_size=1,
            pp_size=1,
            quantization=qconfig,
            use_parallel_embedding=None,
            embedding_sharding_dim=None,
            share_embedding_table=None,
            head_size=-1,  # We need to set it otherwise TRT-LLM tries to compute `hidden_size // num_attention_heads`
            max_source_positions=config.max_source_positions,
            num_mel_bins=config.num_mel_bins,
            trt_model_class="TrtWhisperEncoderPretrainedModel",
            trt_model_file=pathlib.Path(__file__),
        )

        trt_config.mapping.gpus_per_node = min(trt_config.mapping.world_size, 8)

        return trt_config

    def get_plugins_config(self) -> PluginConfig:
        config = super().get_plugins_config()
        config.bert_attention_plugin = self.dtype
        # config.bert_attention_plugin = "disable"
        config.gpt_attention_plugin = "disable"
        config.remove_input_padding = False  # This one is bugged with Whisper.
        config.paged_kv_cache = "disable"  # TODO: getting AssertionError: Paged kv cache is enabled, the kv_cache_block_pointers tensor shall not be None

        config.moe_plugin = "disable"
        config.gemm_plugin = self.dtype
        config.context_fmha = True
        config.enable_xqa = False
        config.remove_input_padding = False
        config.use_custom_all_reduce = "disable"

        config.layernorm_quantization_plugin = None
        config.rmsnorm_quantization_plugin = None
        config.nccl_plugin = None
        # config.paged_kv_cache = False

        return config

    @staticmethod
    def supports_strong_typing() -> bool:
        return False


class WhisperDecoderConfig(TensorRTConfig):
    @classmethod
    def from_config(cls, config: TransformersPretrainedConfig) -> "TensorRTConfig":
        # Retrieve the quantization from the transformers config (if provided)
        _, qconfig = TensorRTConfig.get_quantization_config(config)

        trt_config = cls(
            architecture=config.architectures[0],
            dtype=dtype_to_str(config.torch_dtype),  # TODO: always float32?
            logits_dtype="float32",
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_target_positions,
            hidden_size=config.d_model,
            num_hidden_layers=config.decoder_layers,
            num_attention_heads=config.decoder_attention_heads,
            num_key_value_heads=config.decoder_attention_heads,
            hidden_act=config.activation_function,
            intermediate_size=None,
            norm_epsilon=None,
            position_embedding_type="learned_absolute",
            world_size=1,
            tp_size=1,
            pp_size=1,
            quantization=qconfig,
            use_parallel_embedding=None,
            embedding_sharding_dim=None,
            share_embedding_table=None,
            head_size=-1,  # We need to set it otherwise TRT-LLM tries to compute `hidden_size // num_attention_heads`
            max_source_positions=config.max_source_positions,
            decoder_ffn_dim=config.decoder_ffn_dim,
            trt_model_class="TrtWhisperDecoderPretrainedModel",
            trt_model_file=pathlib.Path(__file__),
            num_encoder_attention_heads=config.encoder_attention_heads,
        )

        trt_config.mapping.gpus_per_node = min(trt_config.mapping.world_size, 8)

        return trt_config

    def get_plugins_config(self) -> PluginConfig:
        config = super().get_plugins_config()
        config.bert_attention_plugin = "disable"
        config.gpt_attention_plugin = self.dtype
        config.paged_kv_cache = "disable"  # TODO: getting AssertionError: Paged kv cache is enabled, the kv_cache_block_pointers tensor shall not be None

        config.context_fmha = True
        config.moe_plugin = "disable"
        config.gemm_plugin = self.dtype
        config.remove_input_padding = False
        config.enable_xqa = False

        config.layernorm_quantization_plugin = None
        config.rmsnorm_quantization_plugin = None
        config.nccl_plugin = None
        # config.paged_kv_cache = False
        config.use_custom_all_reduce = "disable"

        return config

    @staticmethod
    def supports_strong_typing() -> bool:
        return False


class TrtWhisperEncoderPretrainedModel(PretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = TrtWhisperEncoder(
            n_mels=config.num_mel_bins,
            n_ctx=config.max_source_positions,
            n_state=config.hidden_size,
            n_head=config.num_attention_heads,
            n_layer=config.num_hidden_layers,
            dtype=str_dtype_to_trt(config.dtype),
        )

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def prepare_inputs(self, max_batch_size=16, **kwargs):
        (x, input_lengths) = self.model.prepare_inputs(max_batch_size=max_batch_size)

        return {"x": x, "input_lengths": input_lengths}


class TrtWhisperDecoderPretrainedModel(PretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = TrtDecoderModel(
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.decoder_ffn_dim,
            encoder_num_heads=config.num_encoder_attention_heads,
            encoder_hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            dtype=str_dtype_to_trt(config.dtype),
            logits_dtype=str_dtype_to_trt(config.logits_dtype),
            max_position_embeddings=config.max_position_embeddings,
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
            hidden_act=config.hidden_act,
            rescale_before_lm_head=False,
            mapping=config.mapping,
        )
        self.config.optimize_network = False  # See utils/patching.py. TODO: remove this once native to TensorRT-LLM.

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def prepare_inputs(
        self,
        max_batch_size,
        max_input_len,
        max_seq_len,
        use_cache,
        max_beam_width: int = 1,
        max_num_tokens: int = None,
        prompt_embedding_table_size: int = 0,
        position_encoding_2d: bool = False,
        max_draft_len: int = 0,
        gather_context_logits: bool = False,
        gather_generation_logits: bool = False,
        lora_target_modules: List[str] = None,
    ):
        if not use_cache:
            raise NotImplementedError("use_cache=False is not implemented for Whisper.")

        (
            input_ids,
            encoder_output,
            position_ids,
            token_type_ids,
            use_cache,
            attention_mask,
            cross_attention_mask,
            last_token_ids,
            kv_cache_params,
            attention_params,
            hidden_states,
            lora_params,
            cross_kv_cache_gen,
            cross_qkv_reuse,
        ) = self.model.prepare_inputs(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_decoder_input_len=max_input_len,
            max_new_tokens=max_seq_len,
            max_encoder_input_len=self.config.max_source_positions,
        )

        return {
            "decoder_input_ids": input_ids,
            "encoder_output": encoder_output,
            "position_ids": position_ids,
            "token_type_ids": token_type_ids,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
            "cross_attention_mask": cross_attention_mask,
            "last_token_ids": last_token_ids,
            "kv_cache_params": kv_cache_params,
            "attention_params": attention_params,
            "hidden_states": hidden_states,
            "lora_params": lora_params,
            "cross_kv_cache_gen": cross_kv_cache_gen,
            "cross_qkv_reuse": cross_qkv_reuse,
        }


class OptimumWhisperEncoder(TensorRTForSpeechSeq2Seq, HuggingFaceHubModel):
    MODEL_CONFIG = WhisperEncoderConfig
    HF_LIBRARY_TARGET_MODEL_CLASS = TransformersWhisperEncoder
    TRT_LLM_TARGET_MODEL_CLASS = TrtWhisperEncoderPretrainedModel

    @staticmethod
    def convert_weights(
        target: PretrainedModel,
        source: TransformersPretrainedModel,
        config: PretrainedConfig,
    ) -> Dict[str, torch.Tensor]:
        if config.quant_mode.has_any_quant():
            raise NotImplementedError("Quantization is not supported yet.")

        return convert_from_hf_whisper_encoder(source, config.mapping, config.dtype)


class OptimumWhisperDecoder(TensorRTForSpeechSeq2Seq, HuggingFaceHubModel):
    MODEL_CONFIG = WhisperDecoderConfig
    HF_LIBRARY_TARGET_MODEL_CLASS = TransformersWhisperDecoder
    TRT_LLM_TARGET_MODEL_CLASS = TrtWhisperDecoderPretrainedModel

    @staticmethod
    def convert_weights(
        target: PretrainedModel,
        source: TransformersPretrainedModel,
        config: PretrainedConfig,
    ) -> Dict[str, torch.Tensor]:
        if config.quant_mode.has_any_quant():
            raise NotImplementedError("Quantization is not supported yet.")

        return convert_from_hf_whisper_decoder(source, config.mapping, config.dtype)


class WhisperForConditionalGeneration(TensorRTForSpeechSeq2Seq, HuggingFaceHubModel):
    MODEL_CONFIG = None
    HF_LIBRARY_TARGET_MODEL_CLASS = TransformersWhisperForConditionalGeneration
    TRT_LLM_TARGET_MODEL_CLASS = None  # Whisper is split in two in TRT-LLM.

    def __init__(
        self,
        engines_folders: List[Path],
        *,
        gpus_per_node: int,
        transformers_config: "PretrainedConfig",
        use_cuda_graph: bool = False,
        generation_config: Optional[GenerationConfig] = None,
    ):
        super().__init__(engines_folders, gpus_per_node=gpus_per_node, transformers_config=transformers_config, use_cuda_graph=use_cuda_graph, generation_config=generation_config)

        if generation_config is None:
            generation_config = GenerationConfig()
        self.generation_config = generation_config

        self.transformers_config = transformers_config

        # Encoder.
        serialize_path = engines_folders[0] / f'rank0.engine'
        with open(serialize_path, 'rb') as f:
            encoder_session = Session.from_serialized_engine(f.read())

        self.encoder_session = encoder_session

        # Decoder.
        decoder_config_path = engines_folders[1] / 'config.json'
        with open(decoder_config_path, 'r') as f:
            decoder_config = json.load(f)

        serialize_path = engines_folders[1] / f'rank0.engine'
        with open(serialize_path, "rb") as f:
            decoder_engine_buffer = f.read()

        build_config = BuildConfig.from_dict(decoder_config["build_config"])
        trt_config = WhisperDecoderConfig.from_dict(decoder_config["pretrained_config"])

        self.dtype = trt_config.dtype

        decoder_model_config = ModelConfig(
            max_batch_size=build_config.max_batch_size,
            max_beam_width=build_config.max_beam_width,
            num_heads=trt_config.num_attention_heads,
            num_kv_heads=trt_config.num_key_value_heads,
            hidden_size=trt_config.hidden_size,
            vocab_size=trt_config.vocab_size,
            num_layers=trt_config.num_hidden_layers,
            gpt_attention_plugin=build_config.plugin_config.gpt_attention_plugin,
            remove_input_padding=build_config.plugin_config.remove_input_padding,
            cross_attention=True,
            has_position_embedding=True,
            has_token_type_embedding=False,
        )

        # world_size > 1 is not supported.
        world_size = 1
        runtime_rank = mpi_rank()

        runtime_mapping = Mapping(world_size, runtime_rank)

        decoder_generation_session = GenerationSession(
            decoder_model_config,
            decoder_engine_buffer,
            runtime_mapping,
            debug_mode=False)

        self.decoder_generation_session = decoder_generation_session
    
    @classmethod
    def convert_and_build(
        cls,
        local_path: Path,
        hf_model_config: Dict,
        engine_save_path: Optional[Path] = None,
        hf_model: Optional[TransformersPretrainedModel] = None,
        **model_kwargs,
    ) -> Path:
        max_memory = get_max_memory()

        if hf_model is None:
            # Allocate required components for quantization
            hf_model = cls.HF_LIBRARY_TARGET_MODEL_CLASS.from_pretrained(
                local_path,
                device_map="auto",
                max_memory=max_memory,
                local_files_only=True,
            )

        if engine_save_path is None:
            engine_save_path = local_path

        LOGGER.info(f"Building Whisper encoder...")
        encoder_engines_folder, encoder_engines_relative_folder = (
            OptimumWhisperEncoder.convert_and_build(
                local_path,
                hf_model_config,
                engine_save_path=Path(engine_save_path, "encoder"),
                hf_model=hf_model.model.encoder,
                config_class=WhisperEncoderConfig,
                **model_kwargs,
            )
        )

        LOGGER.info(f"Building Whisper decoder...")
        decoder_engines_folder, decoder_engines_relative_folder = (
            OptimumWhisperDecoder.convert_and_build(
                local_path,
                hf_model_config,
                engine_save_path=Path(engine_save_path, "decoder"),
                hf_model=hf_model.model.decoder,
                config_class=WhisperDecoderConfig,
                **model_kwargs,
            )
        )

        return [encoder_engines_folder[0], decoder_engines_folder[0]], [
            encoder_engines_relative_folder[0],
            decoder_engines_relative_folder[0],
        ]
