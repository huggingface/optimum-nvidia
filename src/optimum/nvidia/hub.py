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
import re
from abc import ABCMeta, abstractmethod
from logging import getLogger
from os import PathLike, scandir, symlink
from os.path import exists as fexists
from os.path import join as fjoin
from pathlib import Path
from shutil import copyfile, copytree
from typing import (
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Type,
    Union,
)

import torch.cuda
from huggingface_hub import ModelHubMixin, snapshot_download
from huggingface_hub.hub_mixin import T
from tensorrt_llm import __version__ as trtllm_version
from tensorrt_llm.models import PretrainedConfig
from tensorrt_llm.models import PretrainedModel as TrtLlmPreTrainedModel
from transformers import AutoConfig, GenerationConfig
from transformers import PretrainedConfig as TransformersPretraineConfig
from transformers.utils import (
    CONFIG_NAME,
    GENERATION_CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
)

from optimum.nvidia import LIBRARY_NAME
from optimum.nvidia.compression.modelopt import ModelOptRecipe
from optimum.nvidia.export import (
    PATH_FOLDER_ENGINES,
    ExportConfig,
    TensorRTModelConverter,
    Workspace,
    auto_parallel,
)
from optimum.nvidia.lang import DataType
from optimum.nvidia.models import (
    SupportsFromHuggingFace,
    SupportsTransformersConversion,
)
from optimum.nvidia.models.base import SupportFromTrtLlmCheckpoint
from optimum.nvidia.utils import get_user_agent
from optimum.nvidia.utils.nvml import get_device_count, get_device_name
from optimum.utils import NormalizedConfig


ATTR_TRTLLM_ENGINE_FOLDER = "__trtllm_engine_folder__"
FILE_TRTLLM_ENGINE_PATTERN = "rank[0-9]*.engine"
FILE_TRTLLM_CHECKPOINT_PATTERN = "rank[0-9]*.engine"
FILE_LICENSE_NAME = "LICENSE"
HUB_SNAPSHOT_ALLOW_PATTERNS = [
    CONFIG_NAME,
    GENERATION_CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    "*.safetensors",
    FILE_LICENSE_NAME,
]


LOGGER = getLogger()


def folder_list_engines(folder: Path) -> Iterable[Path]:
    if folder.exists():
        return list(folder.glob("*.engine"))

    return []


def folder_list_checkpoints(folder: Path) -> Iterable[Path]:
    checkpoint_candidates = []
    if folder.exists():
        # At this stage we don't know if they are checkpoints or other safetensors files
        re_checkpoint_filename = re.compile(r"rank[0-9]+\.safetensors")
        checkpoint_candidates = list(
            map(
                Path,
                filter(
                    lambda item: re_checkpoint_filename.match(item.name),
                    scandir(folder),
                ),
            )
        )

    return checkpoint_candidates


def get_rank_from_filename(filename: str) -> int:
    name = filename.split(".")[0]

    if name.startswith("rank"):
        return int(name[3:])
    else:
        raise ValueError(f"Unknown filename format {filename} to extract rank from")


def get_trtllm_artifact(
    model_id: str, patterns: List[str], add_default_allow_patterns: bool = True
) -> Path:
    if (local_path := Path(model_id)).exists():
        return local_path

    return Path(
        snapshot_download(
            repo_id=model_id,
            repo_type="model",
            library_name=LIBRARY_NAME,
            library_version=trtllm_version,
            user_agent=get_user_agent(),
            allow_patterns=patterns + HUB_SNAPSHOT_ALLOW_PATTERNS
            if add_default_allow_patterns
            else patterns,
        )
    )


def get_trtllm_checkpoints(model_id: str, device: str, dtype: str):
    if (
        workspace := Workspace.from_hub_cache(model_id, device)
    ).checkpoints_path.exists():
        return workspace.checkpoints_path

    return get_trtllm_artifact(model_id, [f"{device}/{dtype}/**/*.safetensors"])


def get_trtllm_engines(model_id: str, device: str, dtype: str):
    if (workspace := Workspace.from_hub_cache(model_id, device)).engines_path.exists():
        return workspace.engines_path

    return get_trtllm_artifact(
        model_id, [f"{device}/{dtype}/**/{PATH_FOLDER_ENGINES}/*.engine"]
    )


def from_ranked_checkpoints(
    checkpoints_folder: Path, target_class: Type[SupportFromTrtLlmCheckpoint]
) -> Generator["TrtLlmPreTrainedModel", None, None]:
    root = str(checkpoints_folder)
    trtllm_config = PretrainedConfig.from_checkpoint(root)

    for rank in range(trtllm_config.mapping.world_size):
        yield target_class.from_checkpoint(root, rank, trtllm_config)


def from_ranked_hf_model(
    local_hf_model_path: Path,
    config: "TransformersPretraineConfig",
    target_class: Type["TrtLlmPreTrainedModel"],
    export_config: "ExportConfig",
):
    root = str(local_hf_model_path)
    for rank in range(export_config.sharding.world_size):
        # Specify the current model's rank
        export_config.sharding.rank = rank

        ranked_model = target_class.from_hugging_face(
            root,
            dtype=DataType.from_torch(config.torch_dtype).value,
            mapping=export_config.sharding,
            load_by_shard=True,
            use_parallel_embedding=export_config.sharding.world_size > 1,
            share_embedding_table=config.tie_word_embeddings,
        )

        ranked_model.config.mapping.rank = rank
        yield ranked_model


class HuggingFaceHubModel(
    ModelHubMixin,
    library_name=LIBRARY_NAME,
    language=["en"],
    tags=["optimum-nvidia", "trtllm"],
    repo_url="https://github.com/huggingface/optimum-nvidia",
    docs_url="https://huggingface.co/docs/optimum/nvidia_overview",
    metaclass=ABCMeta,
):
    def __init__(self, engines_path: Union[str, PathLike, Path]):
        self._engines_path = Path(engines_path)

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
        use_cuda_graph: bool = False,
        device_map: Optional[str] = None,
        export_config: Optional[ExportConfig] = None,
        quantization_config: Optional[ModelOptRecipe] = None,
        force_export: bool = False,
        export_only: bool = False,
        save_intermediate_checkpoints: bool = False,
    ) -> T:
        if get_device_count() < 1:
            raise ValueError("No GPU detected on this platform")

        device_name = get_device_name(0)[-1]

        if "torch_dtype" in config:
            dtype = config["torch_dtype"]
        elif "pretrained_config" in config and "dtype" in config["pretrained_config"]:
            dtype = config["pretrained_config"]["dtype"]
        else:
            raise RuntimeError("Failed to detect model's dtype")

        # Check if the model_id is not a local path
        local_model_id = Path(model_id)

        engines_folder = checkpoints_folder = None
        engine_files = checkpoint_files = []

        # Check if we have a local path to a model OR a model_id on the hub
        if local_model_id.exists() and local_model_id.is_dir():
            if any(engine_files := list(folder_list_engines(local_model_id))):
                # Looking for parent folder not actual specific engine file
                engines_folder = engine_files[0].parent
                checkpoints_folder = None
            else:
                checkpoint_files = list(folder_list_checkpoints(local_model_id))

                if checkpoint_files:
                    checkpoints_folder = checkpoint_files[0].parent

        else:
            # Look for prebuild TRTLLM Engine
            if not force_export:
                LOGGER.debug(f"Retrieving prebuild engine(s) for device {device_name}")
                engines_folder = get_trtllm_engines(model_id, device_name, dtype)
                engine_files = folder_list_engines(engines_folder)

            # if no engine is found, then just try to locate a checkpoint
            if not engine_files:
                LOGGER.debug(f"Retrieving checkpoint(s) for {device_name}")
                checkpoints_folder = get_trtllm_checkpoints(
                    model_id, device_name, dtype
                )
                checkpoint_files = folder_list_checkpoints(checkpoints_folder)

        # If no checkpoint available, we are good for a full export from the Hugging Face Hub
        if not engine_files:
            LOGGER.info(f"No prebuild engines nor checkpoint were found for {model_id}")

            # Retrieve the snapshot if needed
            if local_model_id.is_dir():
                LOGGER.debug(f"Retrieving model from local folder: {local_model_id}")
                original_checkpoints_path_for_conversion = local_model_id
                workspace = Workspace(local_model_id)
            else:
                LOGGER.debug(
                    f"Retrieving model from snapshot {model_id} on the Hugging Face Hub"
                )
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
                    allow_patterns=HUB_SNAPSHOT_ALLOW_PATTERNS,
                )
                workspace = None

            # Retrieve a proper transformers' config
            config = NormalizedConfig(AutoConfig.for_model(**config))
            generation_config = GenerationConfig.from_pretrained(
                original_checkpoints_path_for_conversion
            )

            # This is required to complain with binding license for derivative work
            if fexists(
                fjoin(original_checkpoints_path_for_conversion, FILE_LICENSE_NAME)
            ):
                licence_path = fjoin(
                    original_checkpoints_path_for_conversion, FILE_LICENSE_NAME
                )
            else:
                licence_path = None

            # If no export config, let's grab a default one
            export_config = export_config or ExportConfig.from_config(config)

            # Handle the device_map
            if device_map and device_map == "auto":
                LOGGER.info("Auto-parallel we will be used")
                export_config = auto_parallel(export_config)

            # Forward everything to the exporter
            if isinstance(cls, SupportsTransformersConversion):
                targets = cls.TRT_LLM_TARGET_MODEL_CLASSES

                if not isinstance(targets, Mapping):
                    targets = {"": targets}

                for idx, (subpart, clazz) in enumerate(targets.items()):
                    LOGGER.info(
                        f"Building {model_id} {subpart} ({idx + 1} / {len(targets)})"
                    )

                    converter = TensorRTModelConverter(
                        model_id, subpart, workspace, licence_path
                    )

                    if quantization_config:
                        hf_model = cls.HF_LIBRARY_TARGET_MODEL_CLASS.from_pretrained(
                            original_checkpoints_path_for_conversion,
                            torch_dtype="auto",
                            device_map="auto",
                        )
                        checkpoints_folder = converter.quantize(
                            hf_model, quantization_config
                        )
                        checkpoints_folder = checkpoints_folder.root
                        checkpoint_files = folder_list_checkpoints(checkpoints_folder)

                        del hf_model
                        torch.cuda.empty_cache()

                    # Artifacts resulting from a build are not stored in the location `snapshot_download`
                    # would use. Instead, it uses `cached_assets_path` to create a specific location which
                    # doesn't mess up with the HF caching system. Use can use `save_pretrained` to store
                    # the build artifact into a snapshot friendly place
                    # If this specific location is found, we don't necessary need to rebuild
                    if force_export or not len(
                        list(converter.workspace.engines_path.glob("*.engine"))
                    ):
                        if checkpoint_files and isinstance(
                            clazz, SupportFromTrtLlmCheckpoint
                        ):
                            ranked_models = from_ranked_checkpoints(
                                checkpoints_folder, clazz
                            )
                        elif isinstance(clazz, SupportsFromHuggingFace):
                            ranked_models = from_ranked_hf_model(
                                original_checkpoints_path_for_conversion,
                                config,
                                clazz,
                                export_config,
                            )
                        else:
                            raise TypeError(f"{clazz} can't convert from HF checkpoint")

                        generation_config = GenerationConfig.from_pretrained(
                            original_checkpoints_path_for_conversion
                        )
                        for ranked_model in ranked_models:
                            if save_intermediate_checkpoints:
                                _ = converter.convert(ranked_model)
                                LOGGER.info(
                                    f"Saved intermediate checkpoints at {converter.workspace.checkpoints_path}"
                                )

                            build_config = export_config.to_builder_config(
                                ranked_model.config.quantization.quant_mode
                            )
                            _ = converter.build(ranked_model, build_config)
                            engines_folder = converter.workspace.engines_path
                            generation_config.save_pretrained(engines_folder)

                        LOGGER.info(
                            f"Saved TensorRT-LLM engines at {converter.workspace.engines_path}"
                        )
                    else:
                        LOGGER.info(
                            f"Found existing engines at {converter.workspace.engines_path}"
                        )
            else:
                raise ValueError(
                    "Model doesn't support Hugging Face transformers conversion, aborting."
                )
        else:
            generation_config = GenerationConfig.from_pretrained(engines_folder)

        return cls(
            engines_path=engines_folder,
            generation_config=generation_config,
            load_engines=not export_only,
        )

    @abstractmethod
    def _save_additional_parcels(self, save_directory: Path):
        raise NotImplementedError()

    def _save_pretrained(self, save_directory: Path) -> None:
        device_name = get_device_name(0)[-1]
        save_directory = save_directory.joinpath(device_name)
        save_directory.mkdir(parents=True, exist_ok=True)

        src_license_file_path = self._engines_path.parent / FILE_LICENSE_NAME
        dst_files = [src_license_file_path] if src_license_file_path.exists() else []
        dst_files += list(self._engines_path.glob("*"))

        for file in dst_files:
            try:
                # Ensure target folder exists anyhow
                save_path = save_directory.joinpath(
                    file.relative_to(self._engines_path.parent)
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)

                if not save_path.exists():
                    # Need target_is_directory on Windows
                    # Windows10 needs elevated privilege for symlink which will raise OSError if not the case
                    # Falling back to copytree in this case
                    symlink(file, save_path)
            except OSError as ose:
                LOGGER.error(
                    f"Failed to create symlink from current engine folder {self._engines_path.parent} to {save_directory}. "
                    "Will default to copy based _save_pretrained",
                    exc_info=ose,
                )

                dst = save_directory.joinpath(
                    file.relative_to(self._engines_path.parent)
                )
                if file.is_dir():
                    copytree(file, dst, symlinks=True)
                elif file:
                    copyfile(file, dst)

        self._save_additional_parcels(save_directory)
