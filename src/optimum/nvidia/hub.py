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

from logging import getLogger
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Type,
    Union,
)

from huggingface_hub import ModelHubMixin, snapshot_download
from huggingface_hub.hub_mixin import T
from tensorrt_llm import BuildConfig
from tensorrt_llm import __version__ as trtllm_version
from transformers import AutoConfig, GenerationConfig
from transformers.utils import (
    CONFIG_NAME,
    GENERATION_CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
)

from optimum.nvidia import LIBRARY_NAME, ExportConfig
from optimum.nvidia.export import (
    PATH_FOLDER_CHECKPOINTS,
    PATH_FOLDER_ENGINES,
    TensorRTModelConverter,
)
from optimum.nvidia.lang import DataType
from optimum.nvidia.models import SupportsTransformersConversion, SupportsFromHuggingFace
from optimum.nvidia.utils import get_user_agent
from optimum.nvidia.utils.nvml import get_device_count, get_device_name
from optimum.utils import NormalizedConfig


ATTR_TRTLLM_ENGINE_FOLDER = "__trtllm_engine_folder__"
FILE_TRTLLM_ENGINE_PATTERN = "rank[0-9]*.engine"
FILE_TRTLLM_CHECKPOINT_PATTERN = "rank[0-9]*.engine"
HUB_SNAPSHOT_ALLOW_PATTERNS = [
    CONFIG_NAME,
    GENERATION_CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    "*.safetensors"
]

LOGGER = getLogger()


def get_trtllm_artifact(model_id: str, patterns: List[str]) -> Path:
    return Path(snapshot_download(
        repo_id=model_id,
        repo_type="model",
        library_name=LIBRARY_NAME,
        library_version=trtllm_version,
        user_agent=get_user_agent(),
        allow_patterns=patterns
    ))


class HuggingFaceHubModel(
    ModelHubMixin,
    library_name=LIBRARY_NAME,
    languages=["python", "c++"],
    tags=["optimum-nvidia", "trtllm"],
    repo_url="https://github.com/huggingface/optimum-nvidia",
    docs_url="https://huggingface.co/docs/optimum/nvidia_overview",
):

    def __init__(self):
        super().__init__()

    @classmethod
    def _from_pretrained(
        cls: Type[T],
        *,
        model_id: str,
        config: Dict,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Optional[Union[str, bool]],
        export_config: Optional[ExportConfig] = None,
        force_export: bool = False,
        use_cuda_graph: bool = False
    ) -> T:
        if get_device_count() < 1:
            raise ValueError("No GPU detected on this platform")

        device_name = get_device_name(0)[-1]
        common_hub_path = f"{device_name}/{config['torch_dtype']}"

        # Look for prebuild TRTLLM Engine
        engine_files = checkpoint_files = []
        if not force_export:
            LOGGER.debug(f"Attempt to retrieve prebuild engine(s) for device {device_name}")
            cached_path = get_trtllm_artifact(model_id, [f"{common_hub_path}/**/{PATH_FOLDER_ENGINES}/*.engine"])

            if (engines_config_path := (cached_path / PATH_FOLDER_ENGINES / "config.json")).exists():
                LOGGER.info(f"Found prebuild engines at {engines_config_path.parent}")
                engine_files = engines_config_path.parent.glob(FILE_TRTLLM_ENGINE_PATTERN)

        # if no engine is found, then just try to locate a checkpoint
        if not engine_files:
            LOGGER.debug(f"Attempt to retrieve preconverted checkpoint(s) for device {device_name}")
            cached_path = get_trtllm_artifact(model_id, [f"{common_hub_path}/**/*.safetensors"])

            if (checkpoints_config_path := (cached_path / PATH_FOLDER_CHECKPOINTS / "config.json")).exists():
                LOGGER.info(f"Found preconverted checkpoints at {checkpoints_config_path.parent}")
                checkpoint_files = checkpoints_config_path.parent.glob(FILE_TRTLLM_CHECKPOINT_PATTERN)

        # If no checkpoint available, we are good for a full export from the Hugging Face Hub
        if not checkpoint_files:
            LOGGER.info(f"No prebuild engines nor checkpoint were found, starting from scratch with {model_id}")

            # Retrieve the snapshot if needed
            original_checkpoints_path_for_conversion = snapshot_download(
                model_id,
                repo_type="model",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                allow_patterns=HUB_SNAPSHOT_ALLOW_PATTERNS
            )

            # Retrieve a proper transformers' config
            config = NormalizedConfig(AutoConfig.for_model(**config))
            generation_config = GenerationConfig.from_pretrained(original_checkpoints_path_for_conversion)

            # If no export config, let's grab a default one
            export_config = export_config or ExportConfig.from_config(config)

            # Forward everything to the exporter
            if isinstance(cls, SupportsTransformersConversion) and isinstance(cls.TRT_LLM_TARGET_MODEL_CLASS, SupportsFromHuggingFace):
                model = cls.TRT_LLM_TARGET_MODEL_CLASS.from_hugging_face(
                    original_checkpoints_path_for_conversion,
                    dtype=DataType.from_torch(config.torch_dtype).value,
                    mapping=export_config.sharding.to_mapping(),
                )
                converter = TensorRTModelConverter()
                build_config = export_config.to_builder_config()

                # checkpoints = converter.convert(model)
                engines = converter.build(model, build_config)

            return cls(
                engines if isinstance(engines, list) else [engines.root],
                gpus_per_node=get_device_count(),
                transformers_config=config,
                use_cuda_graph=use_cuda_graph,
                generation_config=generation_config,
            )

    def _save_pretrained(self, save_directory: Path) -> None:
        raise NotImplementedError()
