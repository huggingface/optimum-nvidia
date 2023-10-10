#  coding=utf-8
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
import os
from fsspec.implementations.local import LocalFileSystem
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import NamedTuple, Optional, Type, Union, Dict

from huggingface_hub import ModelHubMixin, HfFileSystem
from huggingface_hub.hub_mixin import T

from optimum.nvidia.configs import ModelConfig
from optimum.nvidia.weights.hub import get_safetensors_files
from tensorrt_llm import Mapping, Module as TRTModule

from optimum.nvidia.weights import DEFAULT_TRT_LLM_HUB_REVISION, SupportsSafetensors

LOGGER = getLogger(__name__)

# Utility classes to store build information
BuildInfo = NamedTuple("BuildInfo", [("parallel", bool), ("num_parallel_jobs", int)])
DEFAULT_SERIAL_BUILD_INFO = BuildInfo(False, 1)

# Utility classes to store sharding information
ShardingInfo = NamedTuple("ShardingInfo", [("world_size", int), ("num_gpus_per_node", int)])
NO_SHARDING = ShardingInfo(1, 1)


class TRTEngineBuilder(ModelHubMixin):
    """

    """

    @classmethod
    def _from_pretrained(
        cls: Type[T],
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Optional[Union[str, bool]],
        **model_kwargs,
    ) -> T:
        config = model_kwargs.get("config", None)  # TODO: Ensure this is ok
        sharding = model_kwargs.get("sharding", NO_SHARDING)  # Override inferred adapter
        adapter = model_kwargs.get("adapter", None)  # Override inferred adapter

        if adapter is None:
            LOGGER.debug(f"Inferring adapter from config: {config['model_type']}")
            raise NotImplementedError()

        # Handle the loading - Note Safetensors is always preferred
        if os.path.isdir(model_id):  # Can either be a local directory
            LOGGER.debug(f"Loading weights from local directory {model_id}")
            fs = LocalFileSystem()

        else:  # Or a model on the Hub
            LOGGER.debug(f"Loading weights from remote Hugging Face Hub {model_id}")
            fs = HfFileSystem()

        # Check for safetensors preferred serialization format
        if issubclass(adapter, SupportsSafetensors):
            for file in get_safetensors_files(fs, model_id):
                adapter.from_safetensors(os.path.join(model_id, file), sharding, fs)

        return cls(config)

    def __init__(self, config: ModelConfig):
        self._build_info: Optional[BuildInfo] = None
        self._sharding_info: Optional[ShardingInfo] = None

    def enable_parallel_build(self, num_jobs: int = -1) -> "TRTEngineBuilder":
        """

        :param num_jobs:
        :return:
        """
        if self._build_info:
            raise Exception(f"Cannot specify twice building info ({self._build_info}).")

        LOGGER.debug(f"Setting parallel build strategy to use a maximum of {num_jobs} parallel jobs")
        self._build_info = BuildInfo(True, -1)

        return self

    def shard(self, world_size: int, num_gpus_per_node: int) -> "TRTEngineBuilder":
        """

        :param world_size:
        :param num_gpus_per_node:
        :return:
        """
        if self._sharding_info:
            raise Exception(f"Cannot specify twice sharding config ({self._sharding_info})")

        LOGGER.debug(f"Setting sharding strategy to world_size={world_size}, num_gpus_per_node={num_gpus_per_node}")
        self._sharding_info = ShardingInfo(world_size, num_gpus_per_node)

        return self

    def build(self, output_path: PathLike):
        self._sharding_info = self._sharding_info or NO_SHARDING

        for rank in range(self._sharding_info.world_size):
            LOGGER.debug(f"Building engine rank={rank} (world_size={self._sharding_info.world_size})")
            sharding_desc = Mapping(self._sharding_info.world_size, rank, self._sharding_info.num_gpus_per_node)

        raise NotImplementedError()
