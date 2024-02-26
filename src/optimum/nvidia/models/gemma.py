#  coding=utf-8
#  Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import time
from logging import getLogger
from typing import Dict

import numpy as np
import tensorrt_llm
import torch
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm import Mapping, str_dtype_to_torch
from tensorrt_llm._utils import numpy_to_torch, pad_vocab_size, torch_to_numpy
from tensorrt_llm.layers import MoeConfig
from tensorrt_llm.models import PretrainedConfig, PretrainedModel
from tensorrt_llm.models.gemma.model import GemmaForCausalLM as TrtGemmaForCausalLM
from tensorrt_llm.models.gemma.weight import dup_kv_weight, extract_layer_idx, split
from tensorrt_llm.plugin import PluginConfig
from tensorrt_llm.runtime.lora_manager import LoraConfig
from transformers import GemmaForCausalLM as TransformersGemmaForCausalLM
from transformers import PretrainedConfig as TransformersPretrainedConfig
from transformers import PreTrainedModel as TransformersPretrainedModel

from optimum.nvidia import TensorRTConfig
from optimum.nvidia.config import dtype_to_str
from optimum.nvidia.hub import HuggingFaceHubModel
from optimum.nvidia.runtime import CausalLM


LOGGER = getLogger(__name__)


def load_from_hf_gemma(
    tensorrt_llm_llama: "GemmaForCausalLM",
    hf_gemma,
    mapping=Mapping(),
    dtype="bfloat16",
    use_gemm_woq_plugin=True,
    lora_config=LoraConfig(),
):
    tensorrt_llm.logger.info("Loading weights from HF Gemma...")
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_llama, "quant_mode", QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()
    num_kv_heads = tensorrt_llm_llama.config.num_key_value_heads
    mha_mode = num_kv_heads == tensorrt_llm_llama.config.num_attention_heads

    model_params = dict(hf_gemma.named_parameters())
    # concatenate, duplicate and reshape q, k, v -> qkv
    for l in range(hf_gemma.config.num_hidden_layers):
        prefix = f"model.layers.{l}.self_attn."
        q_weight = model_params[prefix + "q_proj.weight"]
        k_weight = model_params[prefix + "k_proj.weight"]
        v_weight = model_params[prefix + "v_proj.weight"]
        if not mha_mode:
            head_size = tensorrt_llm_llama.config.hidden_size // tensorrt_llm_llama.config.num_attention_heads
            if num_kv_heads < mapping.tp_size:
                # duplicate the KV heads up to tensor_parallel
                k_weight = dup_kv_weight(k_weight, num_kv_heads, mapping.tp_size)
                v_weight = dup_kv_weight(v_weight, num_kv_heads, mapping.tp_size)
            assert (k_weight.shape[0] % (mapping.tp_size * head_size)) == 0
            assert (v_weight.shape[0] % (mapping.tp_size * head_size)) == 0
            qkv_weight = [q_weight, k_weight, v_weight]
        else:
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

        model_params[prefix + "qkv_proj.weight"] = qkv_weight

    moe_config = MoeConfig(
        tensorrt_llm_llama.config.moe_num_experts,
        tensorrt_llm_llama.config.moe_top_k,
        tensorrt_llm_llama.config.moe_tp_mode,
        tensorrt_llm_llama.config.moe_normalization_mode,
    )
    # concatenate MoE gated activations & stack experts
    for l in range(hf_gemma.config.num_hidden_layers):
        if not moe_config.has_moe():
            continue

        rank_experts = list(range(moe_config.num_experts))
        if moe_config.tp_mode == moe_config.ParallelismMode.EXPERT_PARALLEL:
            rank_experts = mapping.ep_experts(moe_config.num_experts)
        for suffix in ["w1", "w2", "w3"]:
            model_params[f"model.layers.{l}.block_sparse_moe.experts.{suffix}.weight"] = torch.stack(
                [
                    model_params[f"model.layers.{l}.block_sparse_moe.experts.{expert}.{suffix}.weight"]
                    for expert in rank_experts
                ]
            )

        w3 = model_params[f"model.layers.{l}.block_sparse_moe.experts.w3.weight"]
        w2 = model_params[f"model.layers.{l}.block_sparse_moe.experts.w2.weight"]
        w1 = model_params[f"model.layers.{l}.block_sparse_moe.experts.w1.weight"]
        if moe_config.tp_mode == moe_config.ParallelismMode.TENSOR_PARALLEL:
            w3 = split(w3, mapping.tp_size, mapping.tp_rank, dim=1)
            w2 = split(w2, mapping.tp_size, mapping.tp_rank, dim=2)
            w1 = split(w1, mapping.tp_size, mapping.tp_rank, dim=1)
        # concat w3 and w1 for gated expert
        model_params[f"model.layers.{l}.block_sparse_moe.experts.w3w1.weight"] = torch.concat([w3, w1], dim=-2)
        model_params[f"model.layers.{l}.block_sparse_moe.experts.w2.weight"] = w2

    torch_dtype = str_dtype_to_torch(dtype)
    layers_range = mapping.pp_layers(hf_gemma.config.num_hidden_layers)

    vocab_size = hf_gemma.config.vocab_size
    weights = {}
    for k, v in model_params.items():
        t_dtype = torch_dtype if "block_sparse_moe.gate" not in k else torch.float32
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(t_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(t_dtype).detach().cpu())
        if "model.embed_tokens.weight" in k:
            if lora_config.is_valid and lora_config.embedding_weight is not None:
                v = torch_to_numpy(lora_config.embedding_weight.to(torch_dtype).detach().cpu())
            if hf_gemma.config.tie_word_embeddings:
                # lm_head.weight has the same weights as embedding
                if mapping.is_last_pp_rank():
                    if vocab_size % mapping.tp_size != 0:
                        # padding
                        vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
                        pad_width = vocab_size_padded - vocab_size
                        v = torch.from_numpy(
                            np.pad(v.detach().cpu().numpy(), ((0, pad_width), (0, 0)), "constant", constant_values=0)
                        )
                    # v_ = torch_to_numpy(numpy_to_torch(v) * np.sqrt(2048.0))
                    weights["lm_head.weight"] = split(v, mapping.tp_size, mapping.tp_rank)

            if tensorrt_llm_llama.config.use_parallel_embedding:
                v = split(v, mapping.tp_size, mapping.tp_rank, tensorrt_llm_llama.config.embedding_sharding_dim)
            if mapping.is_first_pp_rank():
                weights["transformer.vocab_embedding.weight"] = torch_to_numpy(
                    numpy_to_torch(v).to(torch.float32) * np.sqrt(2048.0)
                )
        elif "model.norm.weight" in k:
            if mapping.is_last_pp_rank():
                weights["transformer.ln_f.weight"] = torch_to_numpy(numpy_to_torch(v) + 1.0)

        elif "lm_head.weight" in k:
            if mapping.is_last_pp_rank():
                if lora_config.is_valid and lora_config.lm_head_weight is not None:
                    v = torch_to_numpy(lora_config.lm_head_weight.to(torch_dtype).detach().cpu())
                    vocab_size = v.shape[0]
                if vocab_size % mapping.tp_size != 0:
                    # padding
                    vocab_size_padded = tensorrt_llm_llama.lm_head.out_features * mapping.tp_size
                    pad_width = vocab_size_padded - vocab_size
                    v = np.pad(v, ((0, pad_width), (0, 0)), "constant", constant_values=0)

                weights["lm_head.weight"] = split(v, mapping.tp_size, mapping.tp_rank)
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None or int(layer_idx) not in layers_range:
                continue
            idx = int(layer_idx) - layers_range[0]
            if "input_layernorm.weight" in k:
                weights["transformer.layers.{}.input_layernorm.weight".format(idx)] = torch_to_numpy(
                    numpy_to_torch(v) + 1.0
                )
            elif "post_attention_layernorm.weight" in k:
                weights["transformer.layers.{}.post_layernorm.weight".format(idx)] = torch_to_numpy(
                    numpy_to_torch(v) + 1.0
                )

            elif "self_attn.qkv_proj.weight" in k:
                if not mha_mode:
                    assert isinstance(v, list) and len(v) == 3
                    wq = split(v[0], mapping.tp_size, mapping.tp_rank)
                    wk = split(v[1], mapping.tp_size, mapping.tp_rank)
                    wv = split(v[2], mapping.tp_size, mapping.tp_rank)
                    split_v = np.concatenate((wq, wk, wv))
                else:
                    q_emb = v.shape[0] // 3
                    model_emb = v.shape[1]
                    v = v.reshape(3, q_emb, model_emb)
                    split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                    split_v = split_v.reshape(3 * (q_emb // mapping.tp_size), model_emb)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    (
                        processed_torch_weights,
                        torch_weight_scales,
                    ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        numpy_to_torch(v), plugin_weight_only_quant_type
                    )
                    if not use_gemm_woq_plugin:
                        weights["transformer.layers.{}.attention.qkv.weight".format(idx)] = v
                    else:
                        weights["transformer.layers.{}.attention.qkv.weight".format(idx)] = processed_torch_weights

                    weights["transformer.layers.{}.attention.qkv.per_channel_scale".format(idx)] = torch_weight_scales
                else:
                    weights["transformer.layers.{}.attention.qkv.weight".format(idx)] = split_v

            elif "self_attn.o_proj.weight" in k:
                # dst = tensorrt_llm_llama.layers[idx].attention.dense.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    (
                        processed_torch_weights,
                        torch_weight_scales,
                    ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        numpy_to_torch(v), plugin_weight_only_quant_type
                    )
                    if not use_gemm_woq_plugin:
                        weights["transformer.layers.{}.attention.dense.weight".format(idx)] = v
                    else:
                        weights["transformer.layers.{}.attention.dense.weight".format(idx)] = processed_torch_weights

                    weights[
                        "transformer.layers.{}.attention.dense.per_channel_scale".format(idx)
                    ] = torch_weight_scales

                else:
                    weights["transformer.layers.{}.attention.dense.weight".format(idx)] = split_v

            elif "mlp.up_proj.weight" in k:
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    (
                        processed_torch_weights,
                        torch_weight_scales,
                    ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        numpy_to_torch(v), plugin_weight_only_quant_type
                    )

                    if not use_gemm_woq_plugin:
                        weights["transformer.layers.{}.mlp.gate.weight".format(idx)] = v
                    else:
                        weights["transformer.layers.{}.mlp.gate.weight".format(idx)] = processed_torch_weights

                    weights["transformer.layers.{}.mlp.gate.per_channel_scale".format(idx)] = torch_weight_scales
                else:
                    weights["transformer.layers.{}.mlp.gate.weight".format(idx)] = split_v

            elif "mlp.down_proj.weight" in k:
                # dst = tensorrt_llm_llama.layers[idx].mlp.proj.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    (
                        processed_torch_weights,
                        torch_weight_scales,
                    ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        numpy_to_torch(v), plugin_weight_only_quant_type
                    )
                    if not use_gemm_woq_plugin:
                        weights["transformer.layers.{}.mlp.proj.weight".format(idx)] = v
                    else:
                        weights["transformer.layers.{}.mlp.proj.weight".format(idx)] = processed_torch_weights

                    weights["transformer.layers.{}.mlp.proj.per_channel_scale".format(idx)] = torch_weight_scales
                else:
                    weights["transformer.layers.{}.mlp.proj.weight".format(idx)] = split_v
            elif "mlp.gate_proj.weight" in k:
                # dst = tensorrt_llm_llama.layers[idx].mlp.fc.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    (
                        processed_torch_weights,
                        torch_weight_scales,
                    ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        numpy_to_torch(v), plugin_weight_only_quant_type
                    )

                    if not use_gemm_woq_plugin:
                        weights["transformer.layers.{}.mlp.fc.weight".format(idx)] = v
                    else:
                        weights["transformer.layers.{}.mlp.fc.weight".format(idx)] = processed_torch_weights

                    weights["transformer.layers.{}.mlp.fc.per_channel_scale".format(idx)] = torch_weight_scales
                else:
                    # dst.value = np.ascontiguousarray(split_v)
                    weights["transformer.layers.{}.mlp.fc.weight".format(idx)] = split_v
            elif "experts.w2.weight" in k:
                # Note: no need for splitting, it's already been done above
                split_v = v
                if use_weight_only:
                    v = np.ascontiguousarray(np.transpose(split_v, axes=(0, 2, 1)))
                    (
                        processed_torch_weights,
                        torch_weight_scales,
                    ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        numpy_to_torch(v), plugin_weight_only_quant_type
                    )
                    weights["transformer.layers.{}.mlp.experts_weight_2".format(idx)] = processed_torch_weights
                    weights["transformer.layers.{}.mlp.experts_scale_2".format(idx)] = torch_weight_scales

                else:
                    weights["transformer.layers.{}.mlp.experts_weight_2".format(idx)] = v
            elif "experts.w3w1.weight" in k:
                # Note: no need for splitting, it's already been done above
                split_v = v
                if use_weight_only:
                    v = np.ascontiguousarray(np.transpose(split_v, axes=(0, 2, 1)))
                    (
                        processed_torch_weights,
                        torch_weight_scales,
                    ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        numpy_to_torch(v), plugin_weight_only_quant_type
                    )
                    weights["transformer.layers.{}.mlp.experts_weight_1".format(idx)] = processed_torch_weights
                    weights["transformer.layers.{}.mlp.experts_scale_1".format(idx)] = torch_weight_scales

                else:
                    weights["transformer.layers.{}.mlp.experts_weight_1".format(idx)] = v

            elif "block_sparse_moe.gate" in k:
                v = split(v, mapping.tp_size, mapping.tp_rank, dim=-1)
                weights["transformer.layers.{}.mlp.router.weight".format(idx)] = v

    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f"Weights loaded. Total time: {t}")
    return weights


class GemmaConfig(TensorRTConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaGemmaConfig`]. It is used to instantiate an Gemma
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Gemma-7B.

    Configuration objects inherit from [`TensorRTConfig`] and can be used to control the model outputs. Read the
    documentation from [`TensorRTConfig`] for more information.
    """

    @staticmethod
    def from_config(config: TransformersPretrainedConfig) -> "TensorRTConfig":
        # Retrieve the quantization from the transformers config (if provided)
        qmode, qconfig = TensorRTConfig.get_quantization_config(config)

        trt_config = GemmaConfig(
            architecture=config.architectures[0],
            dtype=dtype_to_str(config.torch_dtype),
            logits_dtype="float32",
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            head_size=config.head_dim,
            hidden_act=config.hidden_act,
            intermediate_size=config.intermediate_size,
            norm_epsilon=config.rms_norm_eps,
            position_embedding_type="rope_gpt_neox",
            world_size=1,
            tp_size=1,
            pp_size=1,
            quant_mode=qmode,
            quant_kwargs=qconfig.to_dict(),
            use_prompt_tuning=False,
            use_parallel_embedding=False,
            embedding_sharding_dim=0,
            share_embedding_table=False,
            max_lora_rank=64,
        )

        trt_config.mapping.gpus_per_node = min(trt_config.mapping.world_size, 8)

        return trt_config

    def get_plugins_config(self) -> PluginConfig:
        config = super().get_plugins_config()
        config.moe_plugin = "disable"
        config.bert_attention_plugin = "disable"
        config.gpt_attention_plugin = self.dtype
        config.gemm_plugin = self.dtype

        return config

    @staticmethod
    def supports_strong_typing() -> bool:
        return False


class GemmaForCausalLM(CausalLM, HuggingFaceHubModel):
    MODEL_CONFIG = GemmaConfig
    HF_LIBRARY_TARGET_MODEL_CLASS = TransformersGemmaForCausalLM
    TRT_LLM_TARGET_MODEL_CLASS = TrtGemmaForCausalLM

    @staticmethod
    def convert_weights(
        target: PretrainedModel, source: TransformersPretrainedModel, config: PretrainedConfig
    ) -> Dict[str, torch.Tensor]:
        if config.quant_mode.has_any_quant():
            raise NotImplementedError("Quantization is not supported yet.")

        return load_from_hf_gemma(target, source, config.mapping, config.dtype)


# class LlamaForCausalLM(TensorRTModel):
#     __slots__ = ("_runtime", )
#
#     def __init__(self, config: Dict[str, Any], engines_folder: Path, gpus_per_node: int, use_cuda_graph: bool = False):
#         super().__init__(engines_folder)
#
#         self._runtime = TensorRTForCausalLM(config, engines_folder, gpus_per_node, use_cuda_graph)
#
#     def generate(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         max_new_tokens: int = -1,
#         min_length: int = -1,
#         num_beams: int = 1,
#         temperature: float = 1.0,
#         top_k: int = 50,
#         top_p: float = 1.0,
#         repetition_penalty: float = 0.0,
#         length_penalty: float = 1.0,
#         seed: int = 0,
#         pad_token_id: int = 0,
#         bos_token_id: int = 1,
#         eos_token_id: int = 2,
#     ):
#         return self._runtime.generate(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             max_new_tokens=max_new_tokens,
#             min_length=min_length,
#             num_beams=num_beams,
#             temperature=temperature,
#             top_k=top_k,
#             top_p=top_p,
#             repetition_penalty=repetition_penalty,
#             length_penalty=length_penalty,
#             seed=seed,
#             pad_token_id=pad_token_id,
#             bos_token_id=bos_token_id,
#             eos_token_id=eos_token_id
#         )
