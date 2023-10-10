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
from typing import Tuple, Optional

import numpy as np
from fsspec import AbstractFileSystem

from tensorrt_llm import Mapping as ShardingConfig
from tensorrt_llm.models import LLaMAForCausalLM

from optimum.nvidia.models import ConvertibleModel
from optimum.nvidia.weights import SupportsSafetensors, WeightAdapter
from optimum.nvidia.weights.safetensors import walk as walk_safetensors


LOGGER = getLogger(__name__)


class LlamaWeightAdapter(WeightAdapter, SupportsSafetensors):
    """

    """

    def convert_tensor(self, name: str, tensor: np.array, rank: int) -> Tuple[str, np.array]:
        LOGGER.debug(f"Converting tensor {name} ({tensor.dtype}<{tensor.shape}>)")
        world_size = self._sharding_config.world_size
        return "", []

    @classmethod
    def from_safetensors(
        cls,
        path: PathLike,
        sharding_config: ShardingConfig,
        filesystem: Optional[AbstractFileSystem] = None
    ):
        adapter = cls(sharding_config)
        for name, tensor in walk_safetensors(path, filesystem):
            print(name)


class LLamaForCausalLM(ConvertibleModel):
    ADAPTER: LlamaWeightAdapter
    TARGET = LLaMAForCausalLM



