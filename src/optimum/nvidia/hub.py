import json
from logging import getLogger
from pathlib import Path
from typing import Type, Optional, Union, Dict, Protocol, runtime_checkable, Any

import numpy as np
from huggingface_hub import ModelHubMixin, snapshot_download
from huggingface_hub.hub_mixin import T
from safetensors.numpy import save_file as to_safetensors
from tensorrt_llm.models.modeling_utils import PretrainedModel, PretrainedConfig
from transformers import AutoConfig, PreTrainedModel as TransformersPretrainedModel
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from optimum.nvidia import TensorRTConfig
from optimum.nvidia.builder import LocalEngineBuilder
from optimum.nvidia.builder.config import EngineConfigBuilder, EngineConfig
from optimum.nvidia.utils import get_user_agent

LOGGER = getLogger()


@runtime_checkable
class SupportsTensorrtConversion(Protocol):
    MODEL_CONFIG: Type[TensorRTConfig]
    HF_LIBRARY_TARGET_MODEL_CLASS: Type[ModelHubMixin]
    TRT_LLM_TARGET_MODEL_CLASS: Type[PretrainedModel]

    @staticmethod
    def convert_weights(
        target: PretrainedModel,
        source: TransformersPretrainedModel,
        config: PretrainedConfig
    ) -> Dict[str, np.ndarray]:
        ...


class HuggingFaceHubModel(ModelHubMixin, SupportsTensorrtConversion):
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
        **model_kwargs
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

        # TODO: How to handle if there is already a checkpoint or whatever?
        output_path = Path(".engine")
        if not output_path.exists():
            output_path.mkdir()

        # Convert the config from Hugging Face to something TRTLLM understand
        LOGGER.debug(f"Parsing Hub configuration to TRTLLM compatible one (model_type: {model_config['model_type']}).")
        config = HuggingFaceHubModel.convert_config_to_trtllm(cls, model_config)

        # We now have a TRTLLM compatible config, so let's feed it to the target TRTLLM model to create a checkpoint
        LOGGER.debug("Allocating TRTLLM model to build the checkpoint")
        model = cls.TRT_LLM_TARGET_MODEL_CLASS.from_config(config)

        # Let's retrieve the weights for this model
        # NOTE: We use `snapshot_download` to be able to provide a custom user-agent
        # NOTE: maybe we can do the same with `from_pretrained`
        LOGGER.debug(f"Loading the weights from the Hub ({model_id}@{revision})")
        local_path = HuggingFaceHubModel.retrieve_snapshot_from_hub(
            model_id, revision, cache_dir, force_download, proxies, resume_download, local_files_only, token
        )

        # Load the weights
        LOGGER.debug(f"Loading weights from {local_path} into the model ({cls.HF_LIBRARY_TARGET_MODEL_CLASS.__name__})")
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

            # Bind the converted weights against the TRTLLM model
            model.load(converted_weights)

            # Write ranked-checkpoints
            to_safetensors(converted_weights, output_path / f"rank{config.mapping.rank}.safetensors")

        # Write global config
        with open(output_path / "config.json", "w") as config_f:
            json.dump(config.to_dict(), config_f)

        # Retrieve the parameters for building the engine
        if "engine_config" in model_kwargs:
            engine_config = model_kwargs.pop("engine_config")
        else:
            builder = EngineConfigBuilder(config)
            builder.with_inference_profile(
                model_kwargs.pop("max_batch_size", 1),
                model_kwargs.pop("max_prompt_length", 128),
                model_kwargs.pop("max_output_length", model_config["max_position_embeddings"])
            ).with_generation_profile(
                model_kwargs.pop("num_beams", 1),
                model_kwargs.pop("max_new_tokens", 256)
            ).logits_as(model_kwargs.pop("logits_dtype", "float32"))

            if model_kwargs.pop("strongly_typed", False):
                builder.strongly_typed()

            if "max_speculated_draft_length" in model_kwargs:
                builder.with_speculated_decoding(model_kwargs.pop("max_speculated_draft_length"))

            engine_config = builder.build()

        engine_builder = LocalEngineBuilder(config, output_path)
        engine_builder.build(engine_config)

        print()

    @staticmethod
    def convert_config_to_trtllm(cls: Type[SupportsTensorrtConversion], config: Dict[str, Any]) -> PretrainedConfig:
        """
        Convert a configuration initially generated by various Hugging Face libraries like transformers, diffusers, etc.
        to TensorRT-LLM `tensorrt_llm.modeling_utils.PretrainedConfig`.
        :param cls: The target class to allocate the model
        :param config: The original library configuration file
        :return: `tensorrt_llm.modeling_utils.PretrainedConfig`
        """
        config = cls.MODEL_CONFIG.from_config(AutoConfig.for_model(**config))
        if hasattr(config, "check_config"):
            config.check_config()

        return config

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
        :return:
        """
        local_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=["config.json", "*.safetensors", SAFE_WEIGHTS_INDEX_NAME],
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            user_agent=get_user_agent(),
        )

        return Path(local_path)