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
from os import PathLike
from typing import List, Mapping, Set, Tuple, Union

import numpy as np

from optimum.nvidia.configs import ModelConfig
from optimum.nvidia.lang import DataType
from optimum.nvidia.models import ConvertibleModel
from optimum.nvidia.weights import WeightAdapter
from optimum.nvidia.weights.safetensors import SupportsSafetensors, SafetensorsAccessor
from safetensors import deserialize
from tensorrt_llm import BuilderConfig, Mapping as ShardingConfig, Module
from tensorrt_llm.models import LLaMAForCausalLM


LOGGER = getLogger(__name__)


class LlamaWeightAdapter(WeightAdapter, SupportsSafetensors):
    """

    """

    @property
    def global_weights(self) -> Set[str]:
        return {""}

    def convert(
        self,
        model: Module,
        config: ModelConfig,
        builder: BuilderConfig,
        rank: int,
        weights: Mapping[str, np.array]
    ) -> Module:
        shard = self._sharding_config

        # TODO: Maybe get this outside of llama specifics
        for layer_idx in range(config.num_layers):
            prefix = f"model.layers.{layer_idx}.self_attn."

            # Merge QKV
            q_weight = weights[prefix + 'q_proj.weight']
            k_weight = weights[prefix + 'k_proj.weight']
            v_weight = weights[prefix + 'v_proj.weight']

            if config.use_multi_query_attention:
                head_size = config.hidden_size // config.num_heads
                if config.num_kv_heads < shard.tp_size:
                    LOGGER.debug(f"Dupplicate the KV heads ({config.num_kv_heads}) up to TP-degree ({shard.tp_size})")

                    for weight in (k_weight, v_weight):
                        factor = tp_size // num_head
                        weight = weight.reshape(num_head, 1, head_size, -1).repeat(factor, axis=1)
                        weight = weight.reshape(num_head * reps * head_size, -1).clone()
                    qkv_weight = [q_weight, k_weight, v_weight]
            else:
                qkv_weight = np.concatenate((q_weight, k_weight, v_weight), axis=0)

            # Insert the packed weights inside the weights
            # weights[prefix + "qkv_proj.weight"] = qkv_weight

        dtype = np.dtype(builder.precision)
        layers_per_pipeline_stage = config.num_layers // shard.pp_size
        layers_range = list(
            range(shard.pp_rank * layers_per_pipeline_stage, (shard.pp_rank + 1) * layers_per_pipeline_stage, 1)
        )

        LOGGER.debug(f"Converting LLama with dtype: {dtype} for rank {rank} and layers: {layers_range}")

        # Convert specific tensors
        if shard.is_first_pp_rank():
            embeddings = weights['model.embed_tokens.weight'].astype(dtype)
            if model.use_parallel_embedding:
                embeddings = split(embeddings, shard.tp_size, rank, model.embedding_sharding_dim)

            model.vocab_embedding.weight.value = embeddings

        if shard.is_last_pp_rank():
            # Final layer norm
            final_norm = weights['model.norm.weight'].astype(dtype)
            model.ln_f.weight.value = final_norm

            # Final vocab projection
            lm_head = weights["lm_head.weight"].astype(dtype)
            # TODO: shard()
            # split_tensor = split(lm_head, shard.tp_size, rank)
            # model.lm_head.weight.value = np.ascontiguousarray(split_tensor)
            model.lm_head.weight.value = lm_head

        return model

    @staticmethod
    def allocate_model(config: ModelConfig, sharding: ShardingConfig, dtype: DataType) -> Module:
        LOGGER.debug(f"Allocating {LLaMAForCausalLM.__name__} model")
        return LLaMAForCausalLM(
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            max_position_embeddings=config.max_sequence_length,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            mlp_hidden_size=config.intermediate_size,
            hidden_act=config.activation,
            dtype=dtype.value,
            mapping=sharding
        )

    @classmethod
    def from_safetensors(
        cls,
        paths: List[Union[str, PathLike]],
        model: Module,
        config: ModelConfig,
        builder_config: BuilderConfig,
        sharding_config: ShardingConfig,
    ) -> Module:
        if not isinstance(model, LLaMAForCausalLM):
            raise ValueError(f"model has to be a derived type from LLaMAForCausalLM, got {type(model)}")

        accessor = SafetensorsAccessor.from_files(paths)
        adapter = cls(sharding_config)
        adapter.convert(model, config, builder_config, sharding_config.rank, accessor)

        return model

class LLamaForCausalLM(ConvertibleModel):
    ADAPTER: LlamaWeightAdapter
    TARGET = LLaMAForCausalLM



