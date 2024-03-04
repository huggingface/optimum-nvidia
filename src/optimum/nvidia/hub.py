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

import json
import shutil
from glob import glob, iglob
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Set, Tuple, Type, Union, runtime_checkable

import numpy as np
from huggingface_hub import ModelHubMixin, snapshot_download
from huggingface_hub.hub_mixin import T
from safetensors.torch import save_file as to_safetensors
from tensorrt_llm._utils import numpy_to_torch
from tensorrt_llm.models.modeling_utils import PretrainedConfig, PretrainedModel
from transformers import AutoConfig
from transformers import PreTrainedModel as TransformersPretrainedModel
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from optimum.nvidia import TensorRTConfig
from optimum.nvidia.builder import LocalEngineBuilder
from optimum.nvidia.builder.config import EngineConfigBuilder
from optimum.nvidia.utils import get_user_agent


ATTR_TRTLLM_ENGINE_FOLDER = "__trtllm_engine_folder__"
FILE_TRTLLM_ENGINE_PATTERN = "rank[0-9]*.engine"
HUB_TRTLLM_ENGINE_PATTERNS = ["config.json", FILE_TRTLLM_ENGINE_PATTERN]
HUB_SAFETENSORS_PATTERNS = ["config.json", "*.safetensors", SAFE_WEIGHTS_INDEX_NAME]

LOGGER = getLogger()


def extract_model_type(config: Dict[str, Any]) -> Tuple[Optional[str], bool]:
    if "model_type" in config:
        model_type = config["model_type"]
        is_tensorrt_config = False

    # This path try to extract from the TensorRTLLM config
    elif "pretrained_config" in config and "architecture" in config["pretrained_config"]:
        model_type = config["pretrained_config"]["architecture"]
        prefix_pos = model_type.index("For")
        model_type = model_type[:prefix_pos].lower()
        is_tensorrt_config = True
    else:
        return None, False

    return model_type, is_tensorrt_config


def find_prebuilt_engine_files(root: Path) -> Optional[Set[Path]]:
    """
    Attempt to locate any prebuilt TRT engine files at the provided root folder and return the path to each of them.
    :param root: The directory we should look into for the engine files.
    :return: None if no engine was found, set of `Path` otherwise
    """

    # Look for engine file
    if len(engines := glob(FILE_TRTLLM_ENGINE_PATTERN, root_dir=root)):
        return {root.joinpath(engine) for engine in engines}

    return None


@runtime_checkable
class SupportsTensorrtConversion(Protocol):
    MODEL_CONFIG: Type[TensorRTConfig]
    HF_LIBRARY_TARGET_MODEL_CLASS: Type[ModelHubMixin]
    TRT_LLM_TARGET_MODEL_CLASS: Type[PretrainedModel]

    @staticmethod
    def convert_weights(
        target: PretrainedModel, source: TransformersPretrainedModel, config: PretrainedConfig
    ) -> Dict[str, np.ndarray]:
        ...


class HuggingFaceHubModel(ModelHubMixin, SupportsTensorrtConversion):
    @staticmethod
    def convert_and_build():
        pass

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
        if not isinstance(cls, SupportsTensorrtConversion):
            raise ValueError(
                f"{cls} doesn't support converting from Hugging Face Hub model."
                " Please open up an issue at https://github.com/huggingface/optimum-nvidia/issues"
            )

        # Let's make sure we have the config
        model_config = model_kwargs.get("config", None)
        if not model_config:
            raise ValueError(
                "Original model configuration (config.json) was not found."
                "The model configuration is required to build TensorRT-LLM engines."
            )

        # Check if we are using a local path
        if not (local_path := Path(model_id)).exists():
            LOGGER.debug(f"Loading potential prebuilt engines from the Hub ({model_id}@{revision})")

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
        engines_folder = local_path / "engines"
        engines_folder.mkdir(exist_ok=True)

        if find_prebuilt_engine_files(engines_folder) is None:
            model_config = AutoConfig.for_model(**model_config)
            config = HuggingFaceHubModel.convert_config_to_trtllm(cls, model_config)

            LOGGER.info(f"No engine file found in {local_path}, converting and building engines")

            # If local_path exists and is not empty we have a local snapshot
            if not local_path.exists() or len(list(local_path.iterdir())) == 0:
                LOGGER.debug(f"Loading original transformers weights from the Hub ({model_id}@{revision})")

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

            # We now have a TRTLLM compatible config, so let's feed it to the target TRTLLM model to create a checkpoint
            LOGGER.debug("Allocating TRTLLM model to build the checkpoint")
            model = cls.TRT_LLM_TARGET_MODEL_CLASS.from_config(config)

            # Load the weights
            LOGGER.debug(
                f"Loading weights from {local_path} into the model ({cls.HF_LIBRARY_TARGET_MODEL_CLASS.__name__})"
            )
            hf_model = cls.HF_LIBRARY_TARGET_MODEL_CLASS.from_pretrained(
                local_path,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=True,
                token=token,
            )

            # Apply the conversion from Hugging Face weights to TRTLLM
            for rank in range(config.mapping.world_size):
                LOGGER.debug(f"Converting weights from Hugging Face checkpoint {model_id} for rank {rank}")
                config.set_rank(rank)
                converted_weights = cls.convert_weights(model, hf_model, config)
                converted_weights = {name: numpy_to_torch(tensor) for name, tensor in converted_weights.items()}

                # Bind the converted weights against the TRTLLM model
                model.load(converted_weights)

                # Write ranked-checkpoints
                to_safetensors(converted_weights, engines_folder / f"rank{config.mapping.rank}.safetensors")

            # Write global config
            with open(engines_folder / "config.json", "w") as config_f:
                json.dump(config.to_dict(), config_f)

            # Retrieve the parameters for building the engine
            if "engine_config" in model_kwargs:
                engine_config = model_kwargs.pop("engine_config")
            else:
                max_prompt_length = model_kwargs.pop("max_prompt_length", 128)
                max_new_tokens = (
                    model_kwargs.pop("max_output_length", model_config.max_position_embeddings) - max_prompt_length
                )

                if max_new_tokens < 1:
                    raise ValueError(
                        "Unable to build the engine because the generation would lead to max_num_tokens < 1. ("
                        f"max_prompt_length = {max_prompt_length}, "
                        f"max_position_embeddings={model_config.max_position_embeddings}, "
                        f"max_new_tokens={max_new_tokens}"
                        ")"
                    )

                builder = (
                    EngineConfigBuilder(model_config)
                    .with_plugins_config(config.get_plugins_config())
                    .with_inference_profile(model_kwargs.pop("max_batch_size", 1), max_prompt_length, max_new_tokens)
                    .with_generation_profile(
                        model_kwargs.pop("num_beams", 1),
                    )
                    .logits_as(model_kwargs.pop("logits_dtype", "float32"))
                )

                if model_kwargs.pop("strongly_typed", False):
                    builder.strongly_typed()

                if "max_speculated_draft_length" in model_kwargs:
                    builder.with_speculated_decoding(model_kwargs.pop("max_speculated_draft_length"))

                engine_config = builder.build()

            engine_builder = LocalEngineBuilder(config, engines_folder)
            engine_builder.build(engine_config)

        model = cls(
            engines_folder,
            gpus_per_node=model_kwargs.pop("gpus_per_node", 1),
            use_cuda_graph=model_kwargs.pop("use_cuda_graph", False),
        )

        setattr(model, ATTR_TRTLLM_ENGINE_FOLDER, engines_folder)
        return model

    def _save_pretrained(self, save_directory: Path) -> None:
        if not hasattr(self, ATTR_TRTLLM_ENGINE_FOLDER):
            raise ValueError(
                "Unable to determine the root folder containing TensorRT-LLM engines. "
                "Please open-up an issue at https://github.com/huggingface/optimum-nvidia"
            )

        # Retrieve the folder
        engine_folder = Path(getattr(self, ATTR_TRTLLM_ENGINE_FOLDER))

        if engine_folder != save_directory:
            LOGGER.debug(f"Saving engines at {save_directory}")

            save_engine_directory = save_directory / "engines"
            save_engine_directory.mkdir(exist_ok=True)

            # Move the engine(s)
            for engine in iglob(FILE_TRTLLM_ENGINE_PATTERN, root_dir=engine_folder):
                LOGGER.debug(f"Moving file {engine_folder / engine} to {save_engine_directory / engine}")
                shutil.move(engine_folder / engine, save_engine_directory / engine)

            # Move the configuration
            if (config_path := engine_folder / "config.json").exists():
                shutil.move(config_path, save_engine_directory / "config.json")
            else:
                LOGGER.warning(
                    f"No config.json found at {config_path}. It might not be possible to reload the engines."
                )

            if (original_config_path := engine_folder / ".." / "config.json").exists():
                shutil.copy(original_config_path, save_directory)

    @staticmethod
    def convert_config_to_trtllm(
        cls: Type[SupportsTensorrtConversion],
        config: Union[PretrainedConfig, Dict[str, Any]],
    ) -> TensorRTConfig:
        """
        Convert a configuration initially generated by various Hugging Face libraries like transformers, diffusers, etc.
        to TensorRT-LLM `tensorrt_llm.modeling_utils.PretrainedConfig`.
        :param cls: The target class to allocate the model
        :param config: The original library configuration file
        :return: `tensorrt_llm.modeling_utils.PretrainedConfig`
        """
        trt_config = cls.MODEL_CONFIG.from_config(config)
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
        patterns = HUB_TRTLLM_ENGINE_PATTERNS if prebuilt_engines_only else HUB_SAFETENSORS_PATTERNS
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
