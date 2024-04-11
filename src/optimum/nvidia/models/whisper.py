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
import copy
import json
import pathlib
from collections import OrderedDict
from logging import getLogger
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
from tensorrt_llm import Mapping, mpi_rank, str_dtype_to_torch
from tensorrt_llm._utils import str_dtype_to_trt, torch_to_numpy, trt_dtype_to_torch
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.functional import LayerNormPositionType, LayerNormType
from tensorrt_llm.models import DecoderModel as TrtDecoderModel
from tensorrt_llm.models import PretrainedConfig, PretrainedModel
from tensorrt_llm.models import WhisperEncoder as TrtWhisperEncoder
from tensorrt_llm.plugin import PluginConfig
from tensorrt_llm.runtime import GenerationSession, ModelConfig, SamplingConfig
from tensorrt_llm.runtime.generation import LogitsProcessorList
from tensorrt_llm.runtime.session import Session, TensorInfo
from transformers import GenerationConfig
from transformers import PreTrainedModel as TransformersPretrainedModel
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin
from transformers.models.whisper.modeling_whisper import (
    WhisperDecoder as TransformersWhisperDecoder,
)
from transformers.models.whisper.modeling_whisper import (
    WhisperEncoder as TransformersWhisperEncoder,
)
from transformers.models.whisper.modeling_whisper import (
    WhisperForConditionalGeneration as TransformersWhisperForConditionalGeneration,
)
from transformers.models.whisper.tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE

from optimum.nvidia import TensorRTConfig
from optimum.nvidia.config import dtype_to_str
from optimum.nvidia.generation.logits_process import (
    TrtForceTokensLogitsProcessor,
    TrtSuppressTokensAtBeginLogitsProcessor,
    TrtSuppressTokensLogitsProcessor,
    TrtWhisperNoSpeechDetection,
)
from optimum.nvidia.hub import HuggingFaceHubModel
from optimum.nvidia.runtime import TensorRTForSpeechSeq2Seq
from optimum.nvidia.utils.nvml import get_max_memory


if TYPE_CHECKING:
    from transformers import PretrainedConfig as TransformersPretrainedConfig
    from transformers.generation.logits_process import LogitsProcessorList
    from transformers.generation.stopping_criteria import StoppingCriteriaList


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
        model_params["embed_positions.weight"]  # .to(torch_dtype)
    )

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
    def from_config(cls, config: "TransformersPretrainedConfig") -> "TensorRTConfig":
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

        return config

    @staticmethod
    def supports_strong_typing() -> bool:
        return False


class WhisperDecoderConfig(TensorRTConfig):
    @classmethod
    def from_config(cls, config: "TransformersPretrainedConfig") -> "TensorRTConfig":
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


class WhisperForConditionalGeneration(
    TensorRTForSpeechSeq2Seq, HuggingFaceHubModel, WhisperGenerationMixin
):
    MODEL_CONFIG = None
    HF_LIBRARY_TARGET_MODEL_CLASS = TransformersWhisperForConditionalGeneration
    TRT_LLM_TARGET_MODEL_CLASS = None  # Whisper is split in two in TRT-LLM.

    def __init__(
        self,
        engines_folders: List[Path],
        *,
        gpus_per_node: int,
        transformers_config: "TransformersPretrainedConfig",
        use_cuda_graph: bool = False,
        generation_config: Optional[GenerationConfig] = None,
    ):
        super().__init__(
            engines_folders,
            gpus_per_node=gpus_per_node,
            transformers_config=transformers_config,
            use_cuda_graph=use_cuda_graph,
            generation_config=generation_config,
        )

        if generation_config is None:
            generation_config = GenerationConfig()
        self.generation_config = generation_config

        self.config = transformers_config

        # Encoder.
        serialize_path = engines_folders[0] / "rank0.engine"
        with open(serialize_path, "rb") as f:
            encoder_session = Session.from_serialized_engine(f.read())

        self.encoder_session = encoder_session

        # Decoder.
        decoder_config_path = engines_folders[1] / "config.json"
        with open(decoder_config_path, "r") as f:
            decoder_config = json.load(f)

        serialize_path = engines_folders[1] / "rank0.engine"
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
            debug_mode=False,
        )

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

        LOGGER.info("Building Whisper encoder...")
        (
            encoder_engines_folder,
            encoder_engines_relative_folder,
        ) = OptimumWhisperEncoder.convert_and_build(
            local_path,
            hf_model_config,
            engine_save_path=Path(engine_save_path, "encoder"),
            hf_model=hf_model.model.encoder,
            config_class=WhisperEncoderConfig,
            **model_kwargs,
        )

        LOGGER.info("Building Whisper decoder...")
        (
            decoder_engines_folder,
            decoder_engines_relative_folder,
        ) = OptimumWhisperDecoder.convert_and_build(
            local_path,
            hf_model_config,
            engine_save_path=Path(engine_save_path, "decoder"),
            hf_model=hf_model.model.decoder,
            config_class=WhisperDecoderConfig,
            **model_kwargs,
        )

        return [encoder_engines_folder[0], decoder_engines_folder[0]], [
            encoder_engines_relative_folder[0],
            decoder_engines_relative_folder[0],
        ]

    def encoder(
        self,
        input_features: torch.Tensor,
    ):
        if dtype_to_str(input_features.dtype) != self.dtype:
            LOGGER.warning(
                f"input_features should be of dtype {self.dtype}, got {dtype_to_str(input_features.dtype)}. Automatically casting to {self.dtype}."
            )
            input_features = input_features.to(str_dtype_to_torch(self.dtype))

        input_lengths = torch.tensor(
            [input_features.shape[2] // 2 for _ in range(input_features.shape[0])],
            dtype=torch.int32,
            device=input_features.device,
        )

        inputs = OrderedDict()
        inputs["x"] = input_features
        inputs["input_lengths"] = input_lengths

        output_list = [
            TensorInfo("x", str_dtype_to_trt(self.dtype), input_features.shape),
            TensorInfo("input_lengths", str_dtype_to_trt("int32"), input_lengths.shape),
        ]

        output_info = (self.encoder_session).infer_shapes(output_list)

        LOGGER.debug(f"output info {output_info}")
        outputs = {
            t.name: torch.empty(
                tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device="cuda"
            )
            for t in output_info
        }

        stream = torch.cuda.current_stream()

        ok = self.encoder_session.run(
            inputs=inputs, outputs=outputs, stream=stream.cuda_stream
        )

        assert ok, "Engine execution failed"
        stream.synchronize()

        return outputs["output"]

    def _retrieve_logit_processors(
        self, generation_config, logits_processor, begin_index, is_shortform, num_beams
    ):
        # Adapted from WhisperGenerationMixin._retrieve_logit_processors with XxxLogitsProcessor -> TrtXxxLogitsProcessor

        if generation_config.return_timestamps is True:
            # TODO: implement.
            raise NotImplementedError(
                "return_timestamps=True is not implemented with TensorRT-LLM. Please open an issue at https://github.com/huggingface/optimum-nvidia/issues. In the meanwhile, please set `model.generation_config.return_timestamps=False`."
            )

        if generation_config.suppress_tokens is not None:
            suppress_tokens_processor = TrtSuppressTokensLogitsProcessor(
                generation_config.suppress_tokens
            )
            logits_processor = (
                [suppress_tokens_processor]
                if logits_processor is None
                else [suppress_tokens_processor] + logits_processor
            )
            generation_config.suppress_tokens = None

        if generation_config.begin_suppress_tokens is not None:
            begin_suppress_processor = TrtSuppressTokensAtBeginLogitsProcessor(
                generation_config.begin_suppress_tokens, begin_index=begin_index
            )
            logits_processor = (
                [begin_suppress_processor]
                if logits_processor is None
                else [begin_suppress_processor] + logits_processor
            )
            generation_config.begin_suppress_tokens = None

        if generation_config.no_speech_threshold is not None and not is_shortform:
            no_speech_detector = TrtWhisperNoSpeechDetection(
                no_speech_token=generation_config.no_timestamps_token_id - 1,
                begin_index=begin_index,
                scores_is_logprobs=num_beams > 1,
            )
            logits_processor = (
                [no_speech_detector]
                if logits_processor is None
                else [no_speech_detector] + logits_processor
            )
            no_speech_detector.set_model(self)

        if is_shortform and generation_config.forced_decoder_ids is not None:
            forced_tokens_proc = TrtForceTokensLogitsProcessor(
                generation_config.forced_decoder_ids
            )
            # It's important that the `forced_tokens_proc` processor is appended after
            # the suppress_tokens processor or else it might happen that all token logits are suppressed to -inf
            # which would lead to unexpected behavior
            # The better approach here is to NOT make use of the `forced_tokens_proc` for Whisper and instead
            # initialize all of them as `decoder_input_ids`.
            # TODO(Sanchit): Make sure to deprecate this in v4.39 as there will be no `forced_decoder_ids` anymore.
            logits_processor = (
                [forced_tokens_proc]
                if logits_processor is None
                else logits_processor + [forced_tokens_proc]
            )
            generation_config.forced_decoder_ids = None

        return logits_processor

    def _retrieve_init_tokens(
        self, input_features, generation_config, config, num_segment_frames, kwargs
    ):
        # Adapted from WhisperGenerationMixin._retrieve_init_tokens with automatic language detection disabled.

        def replace_or_add(lst: List[int], num: int, itr: Iterator[int]):
            """short function to replace num with a itr in lst"""
            found = any(i in lst for i in itr)
            if found:
                lst = [num if i in itr else i for i in lst]
            else:
                lst.append(num)
            return lst

        task = getattr(generation_config, "task", None)
        language = getattr(generation_config, "language", None)

        if kwargs.get("forced_decoder_ids", None) is not None:
            forced_decoder_ids = kwargs["forced_decoder_ids"]
        elif (
            hasattr(generation_config, "forced_decoder_ids")
            and generation_config.forced_decoder_ids is not None
        ):
            forced_decoder_ids = generation_config.forced_decoder_ids

            if language is None and task is None and forced_decoder_ids[0][1] is None:
                LOGGER.warning_once(
                    "Due to a bug fix in https://github.com/huggingface/transformers/pull/28687 transcription using a multilingual Whisper will default to language detection followed by transcription instead of translation to English."
                    "This might be a breaking change for your use case. If you want to instead always translate your audio to English, make sure to pass `language='en'`."
                )
        elif (
            hasattr(config, "forced_decoder_ids")
            and config.forced_decoder_ids is not None
        ):
            forced_decoder_ids = config.forced_decoder_ids
        else:
            forced_decoder_ids = None

        if forced_decoder_ids is not None and task is not None:
            LOGGER.info(
                f"You have passed task={task}, but also have set `forced_decoder_ids` to {forced_decoder_ids} which creates a conflict. `forced_decoder_ids` will be ignored in favor of task={task}."
            )
            forced_decoder_ids = None
        elif forced_decoder_ids is not None and language is not None:
            LOGGER.info(
                f"You have passed language={language}, but also have set `forced_decoder_ids` to {forced_decoder_ids} which creates a conflict. `forced_decoder_ids` will be ignored in favor of language={language}."
            )
            forced_decoder_ids = None

        init_tokens = [generation_config.decoder_start_token_id]
        if forced_decoder_ids is not None and forced_decoder_ids[0][0] == 1:
            i = 1
            while len(forced_decoder_ids) > 0 and forced_decoder_ids[0][0] == i:
                init_tokens += [forced_decoder_ids[0][1]]
                forced_decoder_ids = forced_decoder_ids[1:]
                i += 1

            if len(forced_decoder_ids) > 0:
                raise ValueError(
                    f"You are using token ids in `forced_decoder_ids` that do not seem to correctly follow the prompt pattern of Whisper. Make sure that {forced_decoder_ids} has an entry for all indices >= 1 and < {forced_decoder_ids[0][0]}.",
                )

        # from v4.39 the forced decoder ids are always None in favour of decoder input ids
        generation_config.forced_decoder_ids = None

        is_lang_id_undefined = len(init_tokens) <= 1 or (
            len(init_tokens) > 1 and init_tokens[1] is None
        )
        if language is not None:
            if language in generation_config.lang_to_id.keys():
                language_token = language
            elif language in TO_LANGUAGE_CODE.keys():
                language_token = f"<|{TO_LANGUAGE_CODE[language]}|>"
            elif language in TO_LANGUAGE_CODE.values():
                language_token = f"<|{language}|>"
            else:
                is_language_code = len(language) == 2
                raise ValueError(
                    f"Unsupported language: {language}. Language should be one of:"
                    f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
                )
            if language_token not in generation_config.lang_to_id:
                raise ValueError(
                    f"{language_token} is not supported by this specific model as it is not in the `generation_config.lang_to_id`."
                    "(You should just add it to the generation config)"
                )

            lang_id = generation_config.lang_to_id[language_token]

            # if language is defined it'll overwrite language ids that might have already been defined via the generation_config
            replace_or_add(init_tokens, lang_id, generation_config.lang_to_id.values())
        elif hasattr(generation_config, "lang_to_id") and is_lang_id_undefined:
            raise ValueError(
                f"The language is not specified in the model's generation_config, and automatic language detection is not supported with TensorRT-LLM. Please set e.g. model.generation_config.language = '<|en|>' for English language. Available languages: {generation_config.lang_to_id.keys()}. Please refer to https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/whisper/tokenization_whisper.py#L95 for the languages codes."
            )

        if task is not None:
            if task in TASK_IDS:
                init_tokens.append(generation_config.task_to_id[generation_config.task])
                task_id = generation_config.task_to_id[generation_config.task]

                # if task is defined it'll overwrite task ids that might have already been defined via the generation_config
                replace_or_add(
                    init_tokens, task_id, generation_config.task_to_id.values()
                )
            else:
                raise ValueError(
                    f"The `{task}`task is not supported. The task should be one of `{TASK_IDS}`"
                )
        elif language is not None and hasattr(generation_config, "task_to_id"):
            # if language is defined, but no task id is in `init_tokens`, default to transcribe
            if not any(i in init_tokens for i in generation_config.task_to_id.values()):
                init_tokens.append(generation_config.task_to_id["transcribe"])

        if (
            not generation_config.return_timestamps
            and hasattr(generation_config, "no_timestamps_token_id")
            and init_tokens[-1] != generation_config.no_timestamps_token_id
        ):
            init_tokens.append(generation_config.no_timestamps_token_id)
        elif (
            generation_config.return_timestamps
            and init_tokens[-1] == generation_config.no_timestamps_token_id
        ):
            LOGGER.info(
                "<|notimestamps|> prompt token is removed from generation_config since `return_timestamps` is set to `'True'`."
            )
            init_tokens = init_tokens[:-1]

        # let's make sure we don't pass `None` tokens as prompt tokens
        init_tokens = [t for t in init_tokens if t is not None]

        return init_tokens

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional["LogitsProcessorList"] = None,
        stopping_criteria: Optional["StoppingCriteriaList"] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        synced_gpus: bool = False,
        return_timestamps: Optional[bool] = None,
        task: Optional[str] = None,
        language: Optional[str] = None,
        is_multilingual: Optional[bool] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_condition_type: Optional[str] = None,  # first-segment, all-segments
        condition_on_prev_tokens: Optional[bool] = None,
        temperature: Optional[Union[float, Tuple[float, ...]]] = None,
        compression_ratio_threshold: Optional[float] = None,
        logprob_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = None,
        num_segment_frames: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        time_precision: float = 0.02,
        return_token_timestamps: Optional[bool] = None,
        return_segments: bool = False,
        return_dict_in_generate: Optional[bool] = None,
        **kwargs,
    ):
        if inputs.device.type != "cuda":
            raise ValueError(
                f"TensorRT-LLM only supports inputs on CUDA device. Got: inputs.device = {inputs.device}"
            )

        def raise_unsupported(value: Any, name: str, default: Any = None):
            if value != default:
                raise ValueError(
                    f"TensorRTForSpeechSeq2Seq.generate does not support the argument {name} (got {name}={value}). Please open an issue at https://github.com/huggingface/optimum-nvidia/issues."
                )

        raise_unsupported(stopping_criteria, name="stopping_criteria")
        raise_unsupported(prefix_allowed_tokens_fn, name="prefix_allowed_tokens_fn")
        raise_unsupported(synced_gpus, name="synced_gpus", default=False)
        raise_unsupported(return_timestamps, name="return_timestamps")
        raise_unsupported(task, name="task")
        raise_unsupported(prompt_ids, name="prompt_ids")
        raise_unsupported(prompt_condition_type, name="prompt_condition_type")
        raise_unsupported(temperature, name="temperature")
        raise_unsupported(attention_mask, name="attention_mask")
        raise_unsupported(time_precision, name="time_precision", default=0.02)
        raise_unsupported(return_token_timestamps, name="return_token_timestamps")
        raise_unsupported(return_segments, name="return_segments", default=False)
        raise_unsupported(return_dict_in_generate, name="return_dict_in_generate")

        # 1. copy generation config
        if generation_config is None:
            generation_config = copy.deepcopy(self.generation_config)
        else:
            generation_config = copy.deepcopy(generation_config)

        self._set_language_and_task(
            language=language,
            task=task,
            is_multilingual=is_multilingual,
            generation_config=generation_config,
        )
        self._set_token_ids(
            generation_config=generation_config,
            config=self.config,
            kwargs=kwargs,
        )
        self._set_thresholds_and_condition(
            generation_config=generation_config,
            logprob_threshold=logprob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_prev_tokens=condition_on_prev_tokens,
        )

        num_beams = kwargs.pop("num_beams", generation_config.num_beams)
        input_stride = 1 * 2  # encoder's conv1 stride * encoder's conv2 stride

        batch_size, total_input_frames = self._retrieve_total_input_frames(
            input_features=inputs, input_stride=input_stride, kwargs=kwargs
        )
        num_segment_frames = input_stride * self.config.max_source_positions
        is_shortform = total_input_frames <= num_segment_frames
        if not is_shortform:
            raise ValueError(
                "Whisper TensorRT-LLM implementation only supports short form for now. Please open an issue at https://github.com/huggingface/optimum-nvidia/issues."
            )

        init_tokens = self._retrieve_init_tokens(
            inputs,
            generation_config=generation_config,
            config=self.config,
            num_segment_frames=num_segment_frames,
            kwargs=kwargs,
        )

        begin_index = len(init_tokens)
        logits_processor = self._retrieve_logit_processors(
            generation_config=generation_config,
            logits_processor=logits_processor,
            begin_index=begin_index,  # begin index is index of first generated decoder token
            is_shortform=is_shortform,
            num_beams=kwargs.get("num_beams", 1),
        )
        logits_processor = LogitsProcessorList(logits_processor)

        encoder_outputs = self.encoder(inputs)

        batch_size = inputs.shape[0]
        one_tensor = torch.ones((batch_size, 1), device="cuda", dtype=torch.long)
        decoder_input_ids = torch.cat([t * one_tensor for t in init_tokens], dim=-1)

        max_new_tokens = kwargs.pop("max_new_tokens", generation_config.max_new_tokens)
        if max_new_tokens is None:
            # Transformers' GenerationConfig.max_new_tokens defaults to None.
            if generation_config.max_length is not None:
                max_new_tokens = (
                    generation_config.max_length - decoder_input_ids.shape[1]
                )
            else:
                raise ValueError("Please specifiy the argument `max_new_tokens`.")

        if (
            max_new_tokens + decoder_input_ids.shape[-1]
            > self.config.max_target_positions
        ):
            max_new_tokens = kwargs.get("max_new_tokens", 0)
            raise ValueError(
                f"The length of `decoder_input_ids` equal `prompt_ids` plus special start tokens is {decoder_input_ids.shape[-1]}, and the `max_new_tokens` "
                f"is {max_new_tokens}. Thus, the combined length of "
                f"`decoder_input_ids` and `max_new_tokens` is: {max_new_tokens + decoder_input_ids.shape[-1]}. This exceeds the "
                f"`max_target_positions` of the Whisper model: {self.config.max_target_positions}. "
                "You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, "
                f"so that their combined length is less than {self.config.max_target_positions}."
            )

        encoder_input_lengths = torch.tensor(
            [encoder_outputs.shape[1] for x in range(encoder_outputs.shape[0])],
            dtype=torch.int32,
            device=inputs.device,
        )

        decoder_input_lengths = torch.tensor(
            [decoder_input_ids.shape[-1] for _ in range(decoder_input_ids.shape[0])],
            dtype=torch.int32,
            device="cuda",
        )
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        cross_attention_mask = torch.ones(
            [encoder_outputs.shape[0], 1, encoder_outputs.shape[1]],
            device=inputs.device,
            dtype=torch.int32,
        )

        sampling_config = SamplingConfig(
            end_id=generation_config.eos_token_id,
            pad_id=generation_config.pad_token_id,
            num_beams=num_beams,
        )

        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            beam_width=num_beams,
            encoder_max_input_length=encoder_outputs.shape[1],
        )

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()

        # output_ids of shape [batch_size, beam_width, output_len]
        output_ids = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_input_lengths,
            cross_attention_mask=cross_attention_mask,
            logits_processor=logits_processor,
        )
        torch.cuda.synchronize()

        return output_ids[
            :,
            0,
            : torch.max(self.decoder_generation_session.sequence_length_buffer) + 1,
        ]
