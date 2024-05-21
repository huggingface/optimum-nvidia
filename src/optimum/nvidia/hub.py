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

import shutil
from glob import glob, iglob
from logging import getLogger
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)
from warnings import warn

import numpy as np
import torch
from huggingface_hub import ModelHubMixin, snapshot_download
from huggingface_hub.hub_mixin import T
from safetensors.torch import save_file as to_safetensors
from tensorrt_llm import Mapping
from tensorrt_llm.models.modeling_utils import PretrainedConfig, PretrainedModel
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers import PreTrainedModel as TransformersPretrainedModel
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from optimum.nvidia import TensorRTConfig
from optimum.nvidia.builder import LocalEngineBuilder
from optimum.nvidia.builder.config import EngineConfigBuilder
from optimum.nvidia.quantization import AutoQuantizationConfig
from optimum.nvidia.quantization.ammo import AmmoQuantizer
from optimum.nvidia.utils import get_user_agent, maybe_offload_weights_to_cpu


ATTR_TRTLLM_ENGINE_FOLDER = "__trtllm_engine_folder__"
ATTR_TRTLLM_CHECKPOINT_FOLDER = "__trtllm_checkpoint_folder__"
FOLDER_TRTLLM_CHECKPOINTS = "checkpoints"
FOLDER_TRTLLM_ENGINES = "engines"
FILE_TRTLLM_CHECKPOINT_PATTERN = "rank[0-9]+.safetensors"
FILE_TRTLLM_ENGINE_PATTERN = "rank[0-9]+.engine"

HUB_TRTLLM_ENGINE_PATTERNS = ["**/config.json", f"**/{FILE_TRTLLM_ENGINE_PATTERN}"]
HUB_SAFETENSORS_PATTERNS = ["config.json", "*.safetensors", SAFE_WEIGHTS_INDEX_NAME]

SHARDING_KWARGS = {"tp_size", "pp_size", "world_size", "gpus_per_node", "rank"}
SHARDING_KWARGS_ALIASES = {
    "tp": "tp_size",
    "pp": "pp_size",
}

LOGGER = getLogger()
LOGGER.setLevel(level="INFO")


def extract_model_type(config: Dict[str, Any]) -> Tuple[Optional[str], bool]:
    if "model_type" in config:
        model_type = config["model_type"]
        is_tensorrt_config = False

    # This path try to extract from the TensorRTLLM config
    elif (
        "pretrained_config" in config and "architecture" in config["pretrained_config"]
    ):
        model_type = config["pretrained_config"]["architecture"]
        prefix_pos = model_type.index("For")
        model_type = model_type[:prefix_pos].lower()
        is_tensorrt_config = True
    else:
        return None, False

    return model_type, is_tensorrt_config


def find_prebuilt_engines(root: Path) -> Tuple[List[Path], List[Path]]:
    """
    Attempt to locate any prebuilt TRT engines at the provided root path or root' subfolders.
    :param root: The directory we should look into for the engine files.
    :return: The list of `Path` where engines were found in root (or root/FOLDER_TRTLLM_ENGINES, or root/**/FOLDER_TRTLLM_ENGINES), and the corresponding relative path to root.
    """
    folders = []
    relative_folders = []

    files = glob(Path(root, f"**/{FOLDER_TRTLLM_ENGINES}/rank*.engine").as_posix())
    for file in files:
        file_directory = Path(Path(file).parents[0])
        folders.append(file_directory)
        relative_folders.append(file_directory.relative_to(root))

    files = glob(Path(root, "rank*.engine").as_posix())
    if len(files) > 0:
        folders.append(root)
        relative_folders.append(".")

    files = glob(Path(root, f"{FOLDER_TRTLLM_ENGINES}/rank*.engine").as_posix())
    if len(files) > 0:
        folders.append(Path(root, FOLDER_TRTLLM_ENGINES))
        relative_folders.append(FOLDER_TRTLLM_ENGINES)

    folders = list(dict.fromkeys(folders))
    relative_folders = list(dict.fromkeys(relative_folders))

    # TODO: Handle this properly - we should enforce the directory names and use dicts instead of lists here.
    if len(relative_folders) == 2 and "decoder" in relative_folders[0].as_posix():
        folders.reverse()
        relative_folders.reverse()

    return folders, relative_folders


def get_sharding_info(**kwargs):
    mapping_kwargs = {}

    for name_, value in kwargs.items():
        # Handle aliased args
        if name_ in SHARDING_KWARGS_ALIASES:
            name = SHARDING_KWARGS_ALIASES[name_]
        else:
            name = name_

        # Ensure we are targeting a sharding kwarg
        if name in SHARDING_KWARGS:
            # Check if not already present
            if name in mapping_kwargs and name != name_:
                LOGGER.warning(f"Parameter {name} already defined through {name_}")

            mapping_kwargs[name] = value

    if mapping_kwargs:
        if "world_size" not in mapping_kwargs:
            mapping_kwargs["world_size"] = (
                mapping_kwargs["tp_size"] * mapping_kwargs["pp_size"]
            )
            LOGGER.debug(
                f"Set sharding_info's world_size to {mapping_kwargs['world_size']}"
            )
        return Mapping(**mapping_kwargs)
    else:
        return None


@runtime_checkable
class SupportsTensorrtConversion(Protocol):
    MODEL_CONFIG: Type[TensorRTConfig]
    HF_LIBRARY_TARGET_MODEL_CLASS: Type[ModelHubMixin]

    TRT_LLM_TARGET_MODEL_CLASS: Type[PretrainedModel]

    @staticmethod
    def convert_weights(
        target: PretrainedModel,
        source: TransformersPretrainedModel,
        config: PretrainedConfig,
    ) -> Dict[str, np.ndarray]: ...


class HuggingFaceHubModel(ModelHubMixin, SupportsTensorrtConversion):
    @classmethod
    def convert_and_build(
        cls,
        local_path: Path,
        hf_model_config: Dict,
        engine_save_path: Optional[Path] = None,
        hf_model: Optional[TransformersPretrainedModel] = None,
        config_class: Optional[TensorRTConfig] = None,
        **model_kwargs,
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        :param local_path:
        :param hf_model_config:
        :param model_kwargs:
        :return:
        """

        # Path where will be stored the engines
        root = engine_save_path if engine_save_path else local_path
        checkpoint_folder = root / FOLDER_TRTLLM_CHECKPOINTS
        engines_folder = root / FOLDER_TRTLLM_ENGINES

        # Ensure all the tree exists
        checkpoint_folder.mkdir(exist_ok=True, parents=True)
        engines_folder.mkdir(exist_ok=True, parents=True)

        # Retrieve configuration
        config = AutoConfig.for_model(**hf_model_config)
        if "torch_dtype" in model_kwargs:
            config.torch_dtype = model_kwargs["torch_dtype"]

        # Get sharding information to forward to the config converted
        model_kwargs["mapping"] = get_sharding_info(**model_kwargs)

        # Convert the original config to a model config TRTLLM understands
        model_config = HuggingFaceHubModel.convert_config_to_trtllm(
            cls, config, config_class=config_class, **model_kwargs
        )

        if model_config.has_moe:
            model_config.moe_config.validate()

        # We now have a TRTLLM compatible config, so let's feed it to the target TRTLLM model to create a checkpoint
        LOGGER.debug("Allocating TRTLLM model to build the checkpoint")
        model = cls.TRT_LLM_TARGET_MODEL_CLASS.from_config(model_config)

        # Retrieve the parameters for building the engine
        if "engine_config" in model_kwargs:
            engine_config = model_kwargs.pop("engine_config")
        else:
            builder = EngineConfigBuilder.from_dict(model_config, **model_kwargs)
            builder.with_plugins_config(model_config.get_plugins_config())
            engine_config = builder.build()

        if engine_config.plugins_config is None:
            engine_config.plugins_config = model_config.get_plugins_config()

        if hf_model is None:
            LOGGER.debug(
                f"Loading weights from {local_path} into the model ({cls.HF_LIBRARY_TARGET_MODEL_CLASS.__name__})"
            )

            hf_model = cls.HF_LIBRARY_TARGET_MODEL_CLASS.from_pretrained(
                local_path,
                torch_dtype="auto",
                device_map="cpu",
                local_files_only=True,
            )
        else:
            if not isinstance(hf_model, cls.HF_LIBRARY_TARGET_MODEL_CLASS):
                raise ValueError(
                    f"Expected a {cls.HF_LIBRARY_TARGET_MODEL_CLASS.__name__} model to be provided, but the argument `hf_model` is a {hf_model.__class__.__name__}."
                )

        hf_model = hf_model.eval()
        hf_model = maybe_offload_weights_to_cpu(hf_model)

        # Retrieve potential quantization config (If provided) - follow the transformers parameter's name
        has_qconfig = "quantization_config" in model_kwargs
        use_fp8 = model_kwargs.get("use_fp8", False)

        if has_qconfig or use_fp8:
            LOGGER.debug("About to quantize Hugging Face model")

            if has_qconfig:
                qconfig = model_kwargs.pop("quantization_config")
            elif use_fp8:
                if (
                    candidate_tokenizer_path := engines_folder.parent.joinpath(
                        "tokenizer.json"
                    )
                ).exists():
                    tokenizer_path = candidate_tokenizer_path.parent
                elif "_model_id" in hf_model_config:
                    tokenizer_path = hf_model_config["_model_id"]
                else:
                    raise ValueError(
                        "Unable to determine the tokenizer to use to quantize this model. "
                        "Please provide a complete QuantizationConfig using "
                        "from_pretrained(..., quantization_config=AutoQuantizationConfig.from_description())"
                    )

                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                qconfig = AutoQuantizationConfig.from_description(
                    weight="float8",
                    activation="float8",
                    tokenizer=tokenizer,
                    dataset="c4-new",
                )

                warn(
                    "Converting model to support float8 inference.\n"
                    f"Calibrating model with dataset='c4', split='train', samples={len(qconfig.calibration_dataset)}.\n"
                    "Note: if text generation doesn't meet your expectations, "
                    "you can control the quantization process manually with this API: "
                    "qconfig = AutoQuantizationConfig.from_description(weight='float8', activation='float8', ...) "
                    "forwarding the configuration to .from_pretrained(..., quantization_config=qconfig)"
                )

            hf_quantizer = AmmoQuantizer(
                quantization_config=qconfig,
                artifact_path=checkpoint_folder,
                tensor_parallel_degree=engine_config.sharding_profile.tensor_parallelism,
                pipeline_parallel_degree=engine_config.sharding_profile.pipeline_parallelism,
                export_tensorrt_llm_config=True,
            )

            hf_quantizer.preprocess_model(hf_model, batch_size=1)
            hf_quantizer.postprocess_model(hf_model)

        else:
            # Apply the conversion from Hugging Face weights to TRTLLM
            for rank in range(model_config.mapping.world_size):
                LOGGER.debug(
                    f"Converting weights from Hugging Face checkpoint for rank {rank}"
                )
                model_config.set_rank(rank)
                converted_weights = cls.convert_weights(model, hf_model, model_config)

                # Bind the converted weights against the TRTLLM model
                model.load(converted_weights)

                # Write ranked-checkpoints
                to_safetensors(
                    converted_weights,
                    checkpoint_folder / f"rank{model_config.mapping.rank}.safetensors",
                )

            # Write global config
            model_config.save_pretrained(checkpoint_folder)

        # We are freeing memory used by the HF Model to let the engine build goes forward
        del hf_model
        torch.cuda.empty_cache()

        # Build
        engine_builder = LocalEngineBuilder(
            model_config, checkpoint_folder, engines_folder
        )
        engine_builder.build(engine_config)

        return (
            [checkpoint_folder],
            [engines_folder],
            [engines_folder.relative_to(local_path)],
        )

    @classmethod
    def _from_pretrained(
        cls: Type[T],
        *,
        model_id: str,
        config: Dict[str, Any],
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Optional[Union[str, bool]],
        **model_kwargs,
    ) -> T:
        # Config attributes are being injected also along "config"...
        for key in config.keys():
            if key in model_kwargs:
                model_kwargs.pop(key)

        if not isinstance(cls, SupportsTensorrtConversion):
            raise ValueError(
                f"{cls} doesn't support converting from Hugging Face Hub model."
                " Please open up an issue at https://github.com/huggingface/optimum-nvidia/issues"
            )

        # Check if we are using a local path
        if not (local_path := Path(model_id)).exists():
            LOGGER.debug(
                f"Loading potential prebuilt engines from the Hub ({model_id}@{revision})"
            )

            # Let's retrieve the weights for this model
            # NOTE: We use `snapshot_download` to be able to provide a custom user-agent
            # NOTE: maybe we can do the same with `from_pretrained`
            local_path = HuggingFaceHubModel.retrieve_snapshot_from_hub(
                model_id,
                revision,
                cache_dir,
                force_download,
                proxies,
                resume_download,
                local_files_only,
                token,
                prebuilt_engines_only=True,
            )

        # Look for prebuilt engine files, if none found, we convert and build
        checkpoints_folders, (engines_folders, relative_paths_engines_folders) = (
            None,
            find_prebuilt_engines(local_path),
        )
        if len(engines_folders) == 0:
            LOGGER.info(
                f"No engine file found in {local_path}, converting and building engines"
            )

            # If local_path exists and is not empty we have a local snapshot
            if (
                not local_path.exists()
                or len(list(local_path.glob("*.safetensors"))) == 0
            ):
                LOGGER.debug(
                    f"Loading original transformers weights from the Hub ({model_id}@{revision})"
                )

                local_path = HuggingFaceHubModel.retrieve_snapshot_from_hub(
                    model_id,
                    revision,
                    cache_dir,
                    force_download,
                    proxies,
                    resume_download,
                    local_files_only,
                    token,
                    prebuilt_engines_only=False,
                )

            checkpoint_folders, engines_folders, relative_paths_engines_folders = (
                cls.convert_and_build(local_path, config, **model_kwargs)
            )
        else:
            LOGGER.info(f"Found pre-built engines at: {engines_folders}")

        try:
            generation_config = GenerationConfig.from_pretrained(
                model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
            )
        except OSError:
            generation_config = None

        transformers_config = AutoConfig.for_model(**config)
        model = cls(
            engines_folders,
            gpus_per_node=model_kwargs.pop("gpus_per_node", 1),
            use_cuda_graph=model_kwargs.pop("use_cuda_graph", False),
            generation_config=generation_config,
            transformers_config=transformers_config,
        )

        setattr(model, ATTR_TRTLLM_CHECKPOINT_FOLDER, checkpoints_folders)
        setattr(model, ATTR_TRTLLM_ENGINE_FOLDER, engines_folders)
        model._relative_path_engines_folders = relative_paths_engines_folders
        return model

    def _save_pretrained(self, save_directory: Path) -> None:
        if not hasattr(self, ATTR_TRTLLM_ENGINE_FOLDER):
            raise ValueError(
                "Unable to determine the root folder containing TensorRT-LLM engines. "
                "Please open-up an issue at https://github.com/huggingface/optimum-nvidia"
            )

        self.config.save_pretrained(save_directory)
        if self.generation_config is not None:
            self.generation_config.save_pretrained(save_directory)

        # Retrieve the folder
        engines_folders = getattr(self, ATTR_TRTLLM_ENGINE_FOLDER)
        for i in range(len(engines_folders)):
            engine_folder = Path(engines_folders[i])
            save_subfolder = self._relative_path_engines_folders[i]

            LOGGER.debug(f"Saving engines at {save_directory}")

            save_engine_directory = save_directory / save_subfolder
            save_engine_directory.mkdir(exist_ok=True, parents=True)

            # Move the engine(s)
            for engine in iglob(FILE_TRTLLM_ENGINE_PATTERN, root_dir=engine_folder):
                LOGGER.debug(
                    f"Moving file {engine_folder / engine} to {save_engine_directory / engine}"
                )
                shutil.copyfile(engine_folder / engine, save_engine_directory / engine)

            # Move the configuration
            if (config_path := engine_folder / "config.json").exists():
                shutil.copyfile(config_path, save_engine_directory / "config.json")
            else:
                LOGGER.warning(
                    f"No config.json found at {config_path}. It might not be possible to reload the engines."
                )

            if (original_config_path := engine_folder / ".." / "config.json").exists():
                shutil.copyfile(original_config_path, save_directory / "config.json")

    @staticmethod
    def convert_config_to_trtllm(
        cls: Type[SupportsTensorrtConversion],
        config: Union[PretrainedConfig, Dict[str, Any]],
        config_class: Optional[TensorRTConfig] = None,
        **additional_params,
    ) -> TensorRTConfig:
        """
        Convert a configuration initially generated by various Hugging Face libraries like transformers, diffusers, etc.
        to TensorRT-LLM `tensorrt_llm.modeling_utils.PretrainedConfig`.
        :param cls: The target class to allocate the model
        :param config: The original library configuration file
        :param config_class: An optional custom TensorRTConfig subclass to load into.
        :param additional_params:
        :return: `tensorrt_llm.modeling_utils.PretrainedConfig`
        """
        if config_class is None:
            config_class = cls.MODEL_CONFIG

        mapping = additional_params.get("mapping", None)

        trt_config = config_class.from_config(config, mapping)
        if hasattr(trt_config, "check_config"):
            trt_config.check_config()

        return trt_config

    @staticmethod
    def retrieve_snapshot_from_hub(
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Optional[Union[str, bool]],
        prebuilt_engines_only: bool = False,
    ) -> Path:
        """

        :param model_id:
        :param revision:
        :param cache_dir:
        :param force_download:
        :param proxies:
        :param resume_download:
        :param local_files_only:
        :param token:
        :param prebuilt_engines_only:
        :return:
        """
        patterns = (
            HUB_TRTLLM_ENGINE_PATTERNS
            if prebuilt_engines_only
            else HUB_SAFETENSORS_PATTERNS
        )
        patterns += ["config.json"]
        local_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=patterns,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            user_agent=get_user_agent(),
        )

        return Path(local_path)
