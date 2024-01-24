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
from collections import defaultdict
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import List, Mapping, Union

import numpy as np
from tensorrt_llm import BuilderConfig, Module
from tensorrt_llm import Mapping as ShardingConfig
from tensorrt_llm.models import LLaMAForCausalLM
from tensorrt_llm.quantization import QuantMode

from optimum.nvidia import TensorRTForCausalLM
from optimum.nvidia.configs import ModelConfig, QuantizationConfig
from optimum.nvidia.lang import DataType
from optimum.nvidia.models import ConvertibleModel
from optimum.nvidia.weights import SupportsNpz, SupportsSafetensors, WeightAdapter, as_numpy, shard, pack_qkv
from optimum.nvidia.weights.safetensors import SafetensorsAccessor

LOGGER = getLogger(__name__)


# TODO: Why is this in the models/ folder while WeightAdapter is in weights/?
class LlamaWeightAdapter(WeightAdapter, SupportsSafetensors, SupportsNpz):
    """ """

    QUANTIZATION_EXCLUDED_PARAMETERS = {"lm_head"}
    TENSORRT_LLM_MODEL_CLASS = LLaMAForCausalLM
    LAYERS_PREFIX = "model.layers"

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
        qkv_packed_layers = pack_qkv(
            num_layers=config.num_layers,
            layer_prefix=self.LAYERS_PREFIX,
            attn_layer_name="self_attn",
            weights=weights,
            precision=precision,
            use_multi_head_attention=config.use_multi_head_attention,
            num_kv_heads=config.num_kv_heads,
            shard_info=shard_info
        )

        layers_per_stage = config.num_layers // shard_info.pp_size  # TODO: is this working if not exactly divisible?
        layers_range = range(shard_info.pp_rank * layers_per_stage, (shard_info.pp_rank + 1) * layers_per_stage, 1)

        LOGGER.debug(f"Converting LLama with dtype: {precision} for rank {rank} and layers: {layers_range}")

        # Convert specific tensors
        if shard_info.is_first_pp_rank():
            embeddings = as_numpy(weights["model.embed_tokens.weight"], precision)
            if model.use_parallel_embedding:
                embeddings = shard(embeddings, rank, shard_info.tp_size, model.embedding_sharding_dim)

            model.vocab_embedding.weight.value = embeddings

        if shard_info.is_last_pp_rank():
            # Final layer norm
            final_norm = as_numpy(weights["model.norm.weight"], precision)
            model.ln_f.weight.value = final_norm

            # Final vocab projection
            lm_head = as_numpy(weights["lm_head.weight"], precision)
            rank_tensor = shard(lm_head, rank, shard_info.tp_size)
            model.lm_head.weight.value = rank_tensor

        # Map all the hidden layers
        for layer_idx in layers_range:
            idx = layer_idx - shard_info.pp_rank * layers_per_stage
            prefix = f"{self.LAYERS_PREFIX}.{idx}"

            # input_layernorm.weight
            input_ln = as_numpy(weights[f"{prefix}.input_layernorm.weight"], precision)
            model.layers[idx].input_layernorm.weight.value = input_ln

            # post_attention_layernorm.weight
            post_attn_ln = as_numpy(weights[f"{prefix}.post_attention_layernorm.weight"], precision)
            model.layers[idx].post_layernorm.weight.value = post_attn_ln

            # Self attention layer
            qkv_weights, qkv_bias = qkv_packed_layers[idx]
            q_weight, k_weight, v_weight = qkv_weights

            if qkv_bias is not None:
                raise ValueError("TensorRT-LLM's Llama does not support query/key/value projection bias.")

            # Shard
            # TODO: support GQA
            if not config.use_multi_head_attention:
                wq, wk, wv = (
                    shard(q_weight, rank, shard_info.tp_size, axis=0),
                    shard(k_weight, rank, shard_info.tp_size, axis=0),
                    shard(v_weight, rank, shard_info.tp_size, axis=0),
                )

                qkv_weight = np.concatenate((wq, wk, wv), axis=0)
            else:
                qkv_weight = np.stack((q_weight, k_weight, v_weight), axis=0)
                rank_tensor = shard(qkv_weight, rank, shard_info.tp_size, axis=1)
                qkv_weight = rank_tensor.reshape(-1, config.hidden_size)

            model.layers[idx].attention.qkv.weight.value = np.ascontiguousarray(qkv_weight)

            # Common projection logic
            for src, dst, shard_axis in [
                ("self_attn.o_proj.weight", model.layers[idx].attention.dense.weight, 1),
                ("mlp.up_proj.weight", model.layers[idx].mlp.gate.weight, 0),
                ("mlp.down_proj.weight", model.layers[idx].mlp.proj.weight, 1),
                ("mlp.gate_proj.weight", model.layers[idx].mlp.fc.weight, 0),
            ]:
                tensor = as_numpy(weights[f"{prefix}.{src}"], precision)
                rank_tensor = shard(tensor, rank, shard_info.tp_size, shard_axis)
                dst.value = np.ascontiguousarray(rank_tensor)

        return model

    @staticmethod
    def allocate_model(
        config: ModelConfig, sharding: ShardingConfig, dtype: DataType, quant_mode: QuantMode
    ) -> Module:
        LOGGER.debug(f"Allocating {LlamaWeightAdapter.TENSORRT_LLM_MODEL_CLASS.__name__} model")
        return LlamaWeightAdapter.TENSORRT_LLM_MODEL_CLASS(
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            max_position_embeddings=config.max_sequence_length,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            mlp_hidden_size=config.intermediate_size,
            hidden_act=config.activation,
            dtype=dtype.as_trt(),
            mapping=sharding,
            quant_mode=quant_mode,
            rms_norm_eps=config["rms_norm_eps"],
            logits_dtype=DataType.FLOAT32.as_trt(),
            embedding_sharding_dim=1,  # As Meta does
            use_fused_mlp=quant_mode
            == QuantMode(0),  # Disable if quantization for now as it remove one scaling factor
        )

    @staticmethod
    def get_scaling_factors(
        weights: Mapping[str, np.array], num_layers: int, mode: QuantMode
    ) -> Mapping[str, List[np.array]]:
        # yapf: disable
        scaling_factors = defaultdict(list)

        for layer in range(num_layers):
            scaling_factors['qkv_act'].append(max(
                weights[f'_np:layers:{layer}:attention:qkv:q:activation_scaling_factor'].item(),
                weights[f'_np:layers:{layer}:attention:qkv:k:activation_scaling_factor'].item(),
                weights[f'_np:layers:{layer}:attention:qkv:v:activation_scaling_factor'].item()
            ))
            scaling_factors['qkv_weights'].append(max(
                weights[f'_np:layers:{layer}:attention:qkv:q:weights_scaling_factor'].item(),
                weights[f'_np:layers:{layer}:attention:qkv:k:weights_scaling_factor'].item(),
                weights[f'_np:layers:{layer}:attention:qkv:v:weights_scaling_factor'].item()
            ))
            if mode.has_fp8_kv_cache():
                # Not calibrarting KV cache.
                scaling_factors['qkv_output'].append(1.0)
            else:
                # TODO: What happens, were to retrieve the scales?
                pass

            scaling_factors['dense_act'].append(weights[f'_np:layers:{layer}:attention:dense:activation_scaling_factor'].item())
            scaling_factors['dense_weights'].append(weights[f'_np:layers:{layer}:attention:dense:weights_scaling_factor'].item())
            scaling_factors['fc_act'].append(weights[f'_np:layers:{layer}:mlp:fc:activation_scaling_factor'].item())
            scaling_factors['fc_weights'].append(weights[f'_np:layers:{layer}:mlp:fc:weights_scaling_factor'].item())
            scaling_factors['gate_act'].append(weights[f'_np:layers:{layer}:mlp:gate:activation_scaling_factor'].item())
            scaling_factors['gate_weights'].append(weights[f'_np:layers:{layer}:mlp:gate:weights_scaling_factor'].item())
            scaling_factors['proj_act'].append(weights[f'_np:layers:{layer}:mlp:proj:activation_scaling_factor'].item())
            scaling_factors['proj_weights'].append(weights[f'_np:layers:{layer}:mlp:proj:weights_scaling_factor'].item())

        # yapf: enable
        for k, v in scaling_factors.items():
            assert len(v) == num_layers, f"Expect scaling factor {k} of length {num_layers}, got {len(v)}"

        return scaling_factors


class LlamaForCausalLM(ConvertibleModel, TensorRTForCausalLM):
    ADAPTER = LlamaWeightAdapter
    TARGET = LLaMAForCausalLM
