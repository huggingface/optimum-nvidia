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
from tensorrt_llm import Mapping as ShardingConfig, Module
from tensorrt_llm.models import LLaMAForCausalLM


LOGGER = getLogger(__name__)


class LlamaWeightAdapter(WeightAdapter, SupportsSafetensors):
    """

    """

    @property
    def global_weights(self) -> Set[str]:
        return {""}

    def convert(self, model: Module, config: BuilderConfig, rank: int, weights: Mapping[str, np.array]) -> Module:
        world_size = self._sharding_config.world_size
        print(config.num_layers)
        # print(weights["model.layers.7.input_layernorm.weight"])



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
    def from_safetensors(cls, paths: List[Union[str, PathLike]], builder_config, sharding_config: BuilderConfig,
                         model: ShardingConfig) -> Module:
        if not isinstance(model, LLaMAForCausalLM):
            raise ValueError(f"model has to be a derived type from LLaMAForCausalLM, got {type(model)}")

        accessor = SafetensorsAccessor.from_files(paths)
        adapter = cls(sharding_config)
        adapter.convert(model, sharding_config.rank, accessor)

        return model

class LLamaForCausalLM(ConvertibleModel):
    ADAPTER: LlamaWeightAdapter
    TARGET = LLaMAForCausalLM



