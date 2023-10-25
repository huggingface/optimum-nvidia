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
from multiprocessing import Pool
from os import PathLike, sched_getaffinity
from pathlib import Path
from typing import NamedTuple, Optional, Type, Union, Dict, List

from huggingface_hub import ModelHubMixin, HfFileSystem
from huggingface_hub.hub_mixin import T

from optimum.nvidia.configs import ModelConfig, TransformersConfig
from optimum.nvidia.lang import DataType
from optimum.nvidia.utils import ensure_file_exists_locally
from optimum.nvidia.weights import SupportsSafetensors, WeightAdapter
from optimum.nvidia.weights.hub import get_safetensors_files
from tensorrt_llm import Mapping as Shard
from tensorrt_llm.builder import Builder, BuilderConfig
from tensorrt_llm.network import net_guard


LOGGER = getLogger(__name__)

# Utility classes to store build information
BuildInfo = NamedTuple("BuildInfo", [("parallel", bool), ("num_parallel_jobs", int)])
SERIAL_BUILD = BuildInfo(False, 1)

# Utility classes to store sharding information
ShardingInfo = NamedTuple("ShardingInfo", [("world_size", int), ("num_gpus_per_node", int)])
NO_SHARDING = ShardingInfo(1, 1)


def create_unique_engine_name(identifier: str, dtype: str, rank: int) -> str:
    return f"{identifier}_{dtype}_{rank}.engine"


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
        adapter = model_kwargs.get("adapter", None)  # Override inferred adapter

        if adapter is None:
            LOGGER.debug(f"Inferring adapter from config: {config['model_type']}")
            raise NotImplementedError()

        # TODO: Handle more things from the params here
        if config and not isinstance(config, TransformersConfig):
            config = TransformersConfig(config)
        else:
            raise ValueError(f"Unsupported configuration type ({type(config).__name__})")

        return cls(model_id, config, adapter)

    def __init__(self, model_id_or_path: Union[str, PathLike], config: ModelConfig, adapter: Type[WeightAdapter]):
        self._model_id_or_path: Union[str, PathLike] = model_id_or_path
        self._model_config: ModelConfig = config
        self._weight_adapter: Type[WeightAdapter] = adapter

        self._dtype = DataType.FLOAT16
        self._build_info: BuildInfo = SERIAL_BUILD
        self._sharding_info: ShardingInfo = NO_SHARDING

    def enable_parallel_build(self, num_jobs: int = -1) -> "TRTEngineBuilder":
        """

        :param num_jobs:
        :return:
        """
        # if self._build_info:
        #     raise Exception(f"Cannot specify twice building info ({self._build_info}).")

        LOGGER.debug(f"Setting parallel build strategy to use a maximum of {num_jobs} parallel jobs")
        self._build_info = BuildInfo(True, num_jobs)

        return self

    def shard(self, world_size: int, num_gpus_per_node: int) -> "TRTEngineBuilder":
        """

        :param world_size:
        :param num_gpus_per_node:
        :return:
        """
        # if self._sharding_info:
        #     raise Exception(f"Cannot specify twice sharding config ({self._sharding_info})")

        LOGGER.debug(f"Setting sharding strategy to world_size={world_size}, num_gpus_per_node={num_gpus_per_node}")
        self._sharding_info = ShardingInfo(world_size, num_gpus_per_node)

        return self

    def to(self, dtype: DataType) -> "TRTEngineBuilder":
        LOGGER.debug(f"Setting target dtype to {str(dtype)}")
        self._dtype = dtype

        return self

    def build(self, output_path: PathLike) -> PathLike:
        self._sharding_info = self._sharding_info or NO_SHARDING

        output_path = Path(output_path)
        if not output_path.exists():
            output_path.mkdir(parents=True)

        # Handle the loading - Note Safetensors is always preferred
        if os.path.isdir(self._model_id_or_path):  # Can either be a local directory
            LOGGER.debug(f"Loading weights from local directory {self._model_id_or_path}")
            fs = LocalFileSystem()

        else:  # Or a model on the Hub
            LOGGER.debug(f"Loading weights from remote Hugging Face Hub {self._model_id_or_path}")
            fs = HfFileSystem()

        # Check for safetensors preferred serialization format
        local_files = []
        if issubclass(self._weight_adapter, SupportsSafetensors):
            for file in get_safetensors_files(fs, self._model_id_or_path):
                local_filepath = Path(ensure_file_exists_locally(fs, self._model_id_or_path, file))
                local_files.append(local_filepath)
        else:
            raise NotImplementedError("We only support loading from Safetensors checkpoints for now.")

        shards_info = [
            Shard(self._sharding_info.world_size, rank, self._sharding_info.num_gpus_per_node)
            for rank in range(self._sharding_info.world_size)
        ]

        if self._build_info.parallel and self._build_info.num_parallel_jobs > 1:
            build_func = self._build_parallel
        else:
            build_func = self._build_serial

        # Let's build
        build_func(shards_info, local_files, output_path)
        return output_path

    def _build_serial(self, shards_info: List[Shard], weight_files: List[PathLike], output_path: Path):
        LOGGER.debug(f"Building TRT engines sequentially")

        for shard in shards_info:
            self._build_engine_for_rank(shard, weight_files, output_path)

    def _build_parallel(self, shard_info: List[Shard], weight_files: List[PathLike], output_path: Path):
        build_info = self._build_info
        num_jobs = build_info.num_parallel_jobs if build_info.num_parallel_jobs > 1 else sched_getaffinity(0)

        # If there are more CPU cores than rank ... Let's reduce the number of jobs
        if num_jobs > len(shard_info):
            num_jobs = shard_info

        LOGGER.debug(f"Building TRT engines in parallel ({num_jobs} processes)")
        with Pool(num_jobs) as builders:
            for shard in shard_info:
                engines = builders.map(self._build_engine_for_rank, shard, weight_files, output_path)

    def _build_engine_for_rank(self, shard: Shard, weight_files: List[PathLike], output_path: Path):
        LOGGER.debug(f"Building engine rank={shard.rank} (world_size={shard.world_size})")

        print(f"Building engine rank={shard.rank} (world_size={shard.world_size})")

        config = self._model_config
        model = self._weight_adapter.allocate_model(config, shard, self._dtype)
        ranked_engine_name = create_unique_engine_name(config["model_type"], self._dtype.value, shard.rank)

        builder = Builder()
        build_config = builder.create_builder_config(
            precision=self._dtype.value,
            tensor_parallel=shard.world_size,
            **config  # Inject model's config
        )

        if issubclass(self._weight_adapter, SupportsSafetensors):
            self._weight_adapter.from_safetensors(weight_files, model, config, build_config, shard)

        # Let's build the network
        network = builder.create_network()
        network.trt_network.name = ranked_engine_name

        # Enable plugins
        network.plugin_config.set_gpt_attention_plugin(dtype=self._dtype.value)
        network.plugin_config.set_gemm_plugin(dtype=self._dtype.value)

        if shard.world_size > 1:
            LOGGER.debug(f"Enabling NCCL plugin as world_size = ({shard.world_size})")
            network.plugin_config.set_nccl_plugin(dtype=self._dtype.value)

        with net_guard(network):
            network.set_named_parameters(model.named_parameters())

        # Let's build the engine
        _ = builder.build_engine(network, build_config)

        # Store the build config for the master (rank = 0) to avoid writing up multiple times the same thing
        if shard.rank == 0:
            config_path = output_path.joinpath("config.json")
            timings_path = output_path.joinpath("timings.cache")

            # Save the computed timings
            builder.save_timing_cache(build_config, timings_path)

            LOGGER.debug(f"Saved rank 0 timings at {timings_path}")

            # Save builder config holding all the engine specificities
            builder.save_config(build_config, config_path)

            LOGGER.debug(f"Saved engine config at {config_path}")

