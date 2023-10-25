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
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from optimum.nvidia.configs import ModelConfig
from optimum.nvidia.lang import DataType
from tensorrt_llm import Mapping as ShardingConfig, Module


class WeightAdapter(ABC):
    """

    """

    __slots__ = ("_sharding_config", )

    def __init__(self, sharding_config: ShardingConfig):
        self._sharding_config = sharding_config

    @abstractmethod
    def convert(self, model: Module, rank: int) -> Module:
        """

        :param model:
        :param rank:
        :return:
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def allocate_model(conf: ModelConfig, sharding: ShardingConfig, dtype: DataType) -> Module:
        """

        :param conf:
        :param sharding
        :param dtype
        :return:
        """
        raise NotImplementedError()