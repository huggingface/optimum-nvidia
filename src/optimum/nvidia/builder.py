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
import json
import os

import torch
from dataclasses import dataclass
from enum import IntEnum, auto

import numpy as np
from fsspec.implementations.local import LocalFileSystem
from logging import getLogger
from multiprocessing import Pool
from os import PathLike, sched_getaffinity
from pathlib import Path
from typing import NamedTuple, Optional, Type, Union, Dict, List

from huggingface_hub import ModelHubMixin, HfFileSystem, CONFIG_NAME
from huggingface_hub.hub_mixin import T
from transformers import AutoModelForCausalLM

from optimum.nvidia import OPTIMUM_NVIDIA_CONFIG_FILE, TENSORRT_TIMINGS_FILE
from optimum.nvidia.configs import ModelConfig, TransformersConfig, QuantizationConfig
from optimum.nvidia.lang import DataType
from optimum.nvidia.utils import ensure_file_exists_locally
from optimum.nvidia.weights import SupportsSafetensors, WeightAdapter, SupportsNpz
from optimum.nvidia.quantization import Calibration

from tensorrt_llm import Mapping as Shard, graph_rewriting
from tensorrt_llm.builder import Builder
from tensorrt_llm.models import quantize_model
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm._utils import trt_version

from optimum.nvidia.weights.hub import get_safetensors_files

LOGGER = getLogger(__name__)


SM_FP8_SUPPORTED = {89, 90}

# Utility classes to store build information
BuildInfo = NamedTuple("BuildInfo", [("parallel", bool), ("num_parallel_jobs", int)])
SERIAL_BUILD = BuildInfo(False, 1)

# Utility classes to store shape information
OptimizationProfile = NamedTuple("OptimizationProfile", [
    ("max_batch_size", int),
    ("max_prompt_length", int),
    ("max_new_tokens", int),
    ("max_output_length", int)
])


# Utility classes to store sharding information
ShardingInfo = NamedTuple("ShardingInfo", [
    ("tp_degree", int),
    ("pp_degree", int),
    ("world_size", int),
    ("num_gpus_per_node", int)
])
NO_SHARDING = ShardingInfo(1, 1, 1, 1)


def create_unique_engine_name(identifier: str, dtype: str, rank: int, tp_degree: int) -> str:
    return f"{identifier}_{dtype}_tp{tp_degree}_rank{rank}.engine"


def create_npz_calibration_filename(identifier: str, rank: int, tp_degree: int) -> str:
    return f"{identifier}_tp{tp_degree}_rank{rank}.npz"


class FileFormat(IntEnum):
    NUMPY_QUANTIZED = auto()
    SAFETENSORS = auto()


@dataclass
class Weights:
    files: Union[Path, List[Path]]
    format: FileFormat

    @property
    def is_folder(self) -> bool:
        return isinstance(self.files, Path) and self.files.is_dir()

    @property
    def is_list_of_files(self) -> bool:
        return isinstance(self.files, List)


class TensorRTEngineBuilder(ModelHubMixin):
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
        # Model
        self._model_id_or_path: Union[str, PathLike] = model_id_or_path
        self._model_config: ModelConfig = config
        self._weight_adapter: Type[WeightAdapter] = adapter

        # Engine build
        self._dtype = DataType.FLOAT16
        self._build_info: BuildInfo = SERIAL_BUILD
        self._sharding_info: ShardingInfo = NO_SHARDING
        self._optimization_profile: Optional[OptimizationProfile] = None

        # Quantization
        self._quantization_config: Optional[QuantizationConfig] = None
        self._quantization_calibration: Optional[Calibration] = None

        # Sampling
        self._beam_width = -1

    def enable_parallel_build(self, num_jobs: int = -1) -> "TensorRTEngineBuilder":
        """

        :param num_jobs:
        :return:
        """
        # if self._build_info:
        #     raise Exception(f"Cannot specify twice building info ({self._build_info}).")

        LOGGER.debug(f"Setting parallel build strategy to use a maximum of {num_jobs} parallel jobs")
        self._build_info = BuildInfo(True, num_jobs)

        return self

    def to(self, dtype: Union[str, DataType] ) -> "TensorRTEngineBuilder":
        """

        :param dtype:
        :return:
        """
        LOGGER.debug(f"Setting target dtype to {str(dtype)}")
        if isinstance(dtype, str):
            dtype = DataType(dtype)

        self._dtype = dtype

        return self

    def shard(self, tp_degree: int, pp_degree: int, world_size: int, num_gpus_per_node: int) -> "TensorRTEngineBuilder":
        """

        :param tp_degree
        :param pp_degree
        :param world_size:
        :param num_gpus_per_node:
        :return:
        """
        # if self._sharding_info:
        #     raise Exception(f"Cannot specify twice sharding config ({self._sharding_info})")

        LOGGER.debug(f"Setting sharding strategy to world_size={world_size}, num_gpus_per_node={num_gpus_per_node}")
        self._sharding_info = ShardingInfo(tp_degree, pp_degree, world_size, num_gpus_per_node)

        return self

    def with_quantization_profile(
        self,
        config: QuantizationConfig,
        calibration: Optional[Calibration] = None
    ) -> "TensorRTEngineBuilder":
        """

        :param config:
        :param calibration:
        :return:
        """
        if config.mode.has_fp8_qdq() or config.mode.has_fp8_kv_cache():
            from optimum.nvidia.utils.nvml import get_device_compute_capabilities
            compute_capabilities = get_device_compute_capabilities(0)

            if compute_capabilities:
                compute_capabilities_ = compute_capabilities[0] * 10 + compute_capabilities[1]

                if compute_capabilities_ not in SM_FP8_SUPPORTED:
                    raise ValueError(
                        f"float8 is not supported on your device (compute capabilities = {compute_capabilities_}). "
                        f"Please use a device with compute capabilities {SM_FP8_SUPPORTED}"
                    )

        # TODO: validate the calibration is required or not
        self._quantization_config = config
        self._quantization_calibration = calibration

        return self


    def with_generation_profile(
        self,
        max_batch_size: int,
        max_prompt_length: int,
        max_new_tokens: int,
        max_output_length: int = None
    ) -> "TensorRTEngineBuilder":
        if max_output_length is None:
            # TODO: Understand why we can set to a larger value?
            # max_output_length = self._model_config.max_sequence_length
            max_output_length = max_prompt_length + max_new_tokens

        LOGGER.debug(
            f"Defining generation profile: "
            f"max_batch_size={max_batch_size}, "
            f"max_prompt_length={max_prompt_length}, "
            f"max_new_tokens={max_new_tokens}",
            f"max_output_length={max_output_length}"
        )
        self._optimization_profile = OptimizationProfile(
            max_batch_size,
            max_prompt_length,
            max_new_tokens,
            max_output_length
        )

        return self

    def with_sampling_strategy(self, num_beams: int) -> "TensorRTEngineBuilder":
        """

        :param num_beams:
        :return:
        """
        LOGGER.debug(f"Enabling sampling with strategy: num_beams={num_beams}")
        self._beam_width = num_beams
        return self

    def validate(self) -> bool:
        if self._quantization_config is None:
            LOGGER.warning(
                "Quantization descriptor was None, assuming no quantization will be applied. "
                "If you want to change this behaviour, please use TRTEngineBuilder.with_quantization_schema()"
            )
            self._quantization_config = QuantizationConfig(QuantMode(0), 0)

        # Optimization profile
        if self._optimization_profile is None:
            raise ValueError(
                "No optimization profile has been defined, please do set the profile you want this engine "
                "to be optimized for through TRTEngineBuilder.with_optimization_profile()."
            )

        # Ensure ranges are compatible
        optim_profile = self._optimization_profile
        model_config = self._model_config
        for prop, (min_value, max_value) in [
            ("max_batch_size", (1, None)),
            ("max_prompt_length", (1, model_config.max_sequence_length - 1)),
            ("max_new_tokens", (1, model_config.max_sequence_length - 1)),
            ("max_output_length", (
                    optim_profile.max_prompt_length + optim_profile.max_new_tokens,
                    model_config.max_sequence_length
            ))
        ]:
            prop_value = getattr(optim_profile, prop)
            if prop_value < min_value:
                raise ValueError(f"Invalid value ({prop_value}) for {prop}. Needs to be >= {min_value}")

            if max_value is not None and prop_value > max_value:
                raise ValueError(f"Invalid value ({prop_value}) for {prop}. Needs to be <= {max_value}")

        if optim_profile.max_prompt_length + optim_profile.max_new_tokens > model_config.max_sequence_length:
            new_max_new_tokens = model_config.max_sequence_length - optim_profile.max_prompt_length
            LOGGER.warning(
                f"max_prompt_tokens ({optim_profile.max_prompt_length}) + max_new_tokens ({optim_profile.max_new_tokens})"
                f" is longer than model's maximum sequence length ({model_config.max_sequence_length}). "
                f"Truncating the max_new_tokens to {new_max_new_tokens}."
            )

        # Sampling info
        if self._beam_width < 1:
            LOGGER.warning(
                "Sampling strategy was not specified, defaulting to greedy search. "
                "If you want to define another sampling strategy, please use TRTEngineBuilder.with_sampling_strategy()."
            )
            self._beam_width = 1

        return True

    def build(self, output_path: PathLike, optmization_level: int = None) -> PathLike:
        # Sharding info
        sharding = self._sharding_info or NO_SHARDING
        shards_info = [
            Shard(sharding.world_size, rank, sharding.num_gpus_per_node, sharding.tp_degree, sharding.pp_degree)
            for rank in range(sharding.world_size)
        ]

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
        if issubclass(self._weight_adapter, SupportsSafetensors):
            local_files = []
            for file in get_safetensors_files(fs, self._model_id_or_path):
                local_filepath = Path(ensure_file_exists_locally(fs, self._model_id_or_path, file))
                local_files.append(local_filepath)
            files = Weights(local_files, FileFormat.SAFETENSORS)
        else:
            raise NotImplementedError("We only support loading from Safetensors checkpoints for now.")

        # Handle potential need for computing calibration data to quantize the model
        if self._quantization_config and self._quantization_config.has_quantization_step:
            from optimum.nvidia.quantization.ammo import AmmoQuantizer
            LOGGER.debug(
                "Model requires quantization ("
                f"weight only: {self._quantization_config.mode.is_weight_only()}, "
                f"mode: {self._quantization_config.mode}"
                ")"
            )

            # Save quantization artifacts
            calibration_path = output_path.joinpath("calibration")

            # Handle any calibration required for static quantization
            if self._quantization_calibration:
                has_json = has_npz = False
                if calibration_path.exists() and calibration_path.is_dir():
                    calibration_data = os.listdir(calibration_path)
                    if len(calibration_data) == 2:
                        has_json = any(f.endswith(".json") for f in calibration_data)
                        has_npz = any(f.endswith(".npz") for f in calibration_data)

                if not calibration_path.exists() or not (has_json and has_npz):
                    LOGGER.info("Calibrating model ...")

                    # Allocate required components for quantization
                    hf_model = AutoModelForCausalLM.from_pretrained(
                        self._model_id_or_path,
                        device_map="auto",
                        torch_dtype=self._dtype.as_torch()
                    ).to(memory_format=torch.channels_last)

                    quantizer = AmmoQuantizer(hf_model, self._quantization_config, self._dtype, sharding.tp_degree)
                    quantizer.calibrate(self._quantization_calibration)
                    quantizer.save(calibration_path)

                    # Release the memory
                    del hf_model
                    torch.cuda.empty_cache()
                else:
                    LOGGER.info(f"Reusing already precomputed calibration data at {calibration_path}")

            files = [files, Weights(calibration_path, FileFormat.NUMPY_QUANTIZED)]

        if self.validate():
            if self._build_info.parallel and self._build_info.num_parallel_jobs > 1:
                build_func = self._build_parallel
            else:
                build_func = self._build_serial

            # Let's build
            build_func(shards_info, files, output_path, optmization_level)
            return output_path

    def _build_serial(
        self,
        shards_info: List[Shard],
        weights: Union[Weights, List[Weights]],
        output_path: Path,
        opt_level: Optional[int]
    ):
        LOGGER.debug(f"Building TRT engines sequentially")

        for shard in shards_info:
            self._build_engine_for_rank(shard, weights, output_path, opt_level, is_parallel=False)

    def _build_parallel(
        self,
        shard_info: List[Shard],
        weight_files: Union[Weights, List[Weights]],
        output_path: Path,
        opt_level: Optional[int]
    ):
        build_info = self._build_info
        num_jobs = build_info.num_parallel_jobs if build_info.num_parallel_jobs > 1 else sched_getaffinity(0)

        # If there are more CPU cores than rank ... Let's reduce the number of jobs
        if num_jobs > len(shard_info):
            num_jobs = shard_info

        LOGGER.debug(f"Building TRT engines in parallel ({num_jobs} processes)")
        with Pool(num_jobs) as builders:
            for shard in shard_info:
                _ = builders.map(
                    self._build_engine_for_rank,
                    shard, weight_files,
                    output_path,
                    is_parallel=True,
                    opt_level=opt_level
                )

    def _build_engine_for_rank(
        self,
        shard: Shard,
        weights: Union[Weights, List[Weights]],
        output_path: Path,
        opt_level: Optional[int],
        is_parallel: bool
    ):
        LOGGER.debug(f"Building engine rank={shard.rank} (world_size={shard.world_size})")

        config = self._model_config
        qconfig = self._quantization_config

        ranked_engine_name = create_unique_engine_name(
            config["model_type"],
            self._dtype.value,
            shard.rank,
            shard.tp_size
        )

        builder = Builder()
        build_config = builder.create_builder_config(
            name=config["model_type"],
            precision=self._dtype.value,
            fp8=qconfig.mode.has_fp8_qdq(),
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            hidden_act=config.activation,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            num_layers=config.num_layers,
            max_position_embeddings=config.max_sequence_length,
            max_batch_size=self._optimization_profile.max_batch_size,
            max_input_len=self._optimization_profile.max_prompt_length,
            max_output_len=self._optimization_profile.max_output_length,
            max_num_tokens=None,
            max_beam_width=self._beam_width,
            strongly_typed=qconfig.mode.has_fp8_qdq(),
            tensor_parallel=shard.tp_size,
            pipeline_parallel=shard.pp_size,
            parallel_build=is_parallel,
            use_refit=False,
            quant_mode=self._quantization_config.mode,
            opt_level=opt_level,
            huggingface=dict(**config),
            tensorrt=trt_version()
        )

        model = self._weight_adapter.allocate_model(config, shard, self._dtype, qconfig.mode)

        if isinstance(weights, Weights):
            weights = [weights]

        for weight in weights:
            # Handle various loading and conversion methods
            if weight.format == FileFormat.SAFETENSORS and issubclass(self._weight_adapter, SupportsSafetensors):
                LOGGER.debug("Using safetensors as weight provider")
                self._weight_adapter.from_safetensors(weight.files, model, config, build_config, qconfig, shard)

            elif weight.format == FileFormat.NUMPY_QUANTIZED and issubclass(self._weight_adapter, SupportsNpz):
                LOGGER.debug("Using Numpy npz as weight provider (with quantization metadata)")
                calibration_filename = create_npz_calibration_filename(config["model_type"], shard.rank, shard.tp_size)
                qweights = np.load(
                    weight.files.joinpath(calibration_filename),
                    mmap_mode="r",
                    allow_pickle=False
                )

                scales = self._weight_adapter.get_scaling_factors(qweights, config.num_layers, qconfig.mode)
                quantize_model(model, qconfig.mode, quant_scales=scales)
            else:
                raise ValueError(
                    f"Unknown FileFormat {weight.format}. "
                    "Please open up an issue on https://github.com/huggingface/optimum-nvidia/issues"
                )

        # Let's build the network
        network = builder.create_network()
        network.trt_network.name = ranked_engine_name

        # Enable plugins
        network.plugin_config.set_gpt_attention_plugin(dtype=self._dtype.value)
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
        network.plugin_config.set_rmsnorm_plugin(dtype=self._dtype.value)
        network.plugin_config.enable_remove_input_padding()

        # GeMM plugin doesn't support float8
        if not build_config.fp8:
            network.plugin_config.set_gemm_plugin(dtype=self._dtype.value)

        # network.plugin_config.enable_paged_kv_cache(64)

        if shard.world_size > 1:
            LOGGER.debug(f"Enabling NCCL plugin as world_size = ({shard.world_size})")
            network.plugin_config.set_nccl_plugin(dtype=self._dtype.value)

        with net_guard(network):
            network.set_named_parameters(model.named_parameters())
            inputs = model.prepare_inputs(
                max_batch_size=self._optimization_profile.max_batch_size,
                max_input_len=self._optimization_profile.max_prompt_length,
                max_new_tokens=self._optimization_profile.max_new_tokens,
                max_num_tokens=None,
                max_beam_width=self._beam_width,
                use_cache=True
            )

            model(*inputs)

            # to_onnx(network.trt_network, output_path.joinpath("model.onnx"))

        LOGGER.debug("Optimizing network ...")
        graph_rewriting.optimize(network)

        # Let's build the engine
        engine = builder.build_engine(network, build_config)

        # Store the build config for the master (rank = 0) to avoid writing up multiple times the same thing
        if shard.rank == 0:
            hf_config_path = output_path.joinpath(CONFIG_NAME)
            build_config_path = output_path.joinpath(OPTIMUM_NVIDIA_CONFIG_FILE)
            timings_path = output_path.joinpath(TENSORRT_TIMINGS_FILE)

            # Save the model's configuration (mainly to restore with from_pretrained without too much pain)
            with open(hf_config_path, "w", encoding="utf-8") as hf_config_f:
                config_ = config.config if hasattr(config, "config") else config
                json.dump(config_, hf_config_f)
                LOGGER.debug(f"Saved HF model config at {hf_config_path}")

            # Save the computed timings
            builder.save_timing_cache(build_config, str(timings_path))
            LOGGER.debug(f"Saved rank 0 timings at {timings_path}")

            # Save builder config holding all the engine specificities
            builder.save_config(build_config, str(build_config_path))
            LOGGER.debug(f"Saved engine config at {build_config_path}")

        self._serialize_engine(engine, output_path.joinpath(ranked_engine_name))

    def _serialize_engine(self, engine, path: Path):
        LOGGER.info(f'Saving engine to {path}...')
        with open(path, 'wb') as f:
            f.write(bytearray(engine))

