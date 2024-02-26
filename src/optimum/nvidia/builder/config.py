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
from typing import Optional

from dataclasses import dataclass
from tensorrt_llm.plugin import PluginConfig
from transformers import PretrainedConfig as TransformersPretrainedConfig


LOGGER = getLogger()
SUPPORTED_LOGITS_DTYPE = {"float32", "float16"}


@dataclass
class InferenceProfile:
    max_batch_size: int
    max_input_len: int
    max_output_len: int


@dataclass
class GenerationProfile:
    num_beams: int
    max_draft_length: int


@dataclass
class EngineConfig:
    """
    Represent all the parameters required to tune and build the final TRTLLM engine(s)
    """
    optimisation_level: int
    strongly_typed: bool
    logits_dtype: str
    workload_profile: InferenceProfile
    generation_profile: GenerationProfile
    plugins_config: PluginConfig



class EngineConfigBuilder:

    def __init__(self, config: TransformersPretrainedConfig):
        self._config = config

        self._optimisation_level: int = 3
        self._strongly_typed: bool = False
        self._workload_profile: Optional[InferenceProfile] = None
        self._generation_profile: Optional[GenerationProfile] = None
        self._plugin_config: Optional[PluginConfig] = None

    def strongly_typed(self) -> "EngineConfigBuilder":
        self._strongly_typed = True
        LOGGER.info(f"Defined engine as strongly typed")
        return self

    def with_optimisation_level(self, level: int) -> "EngineConfigBuilder":
        if level < 1:
            raise ValueError(f"level should be >= 1 (got: {level})")
        self._optimisation_level = level
        LOGGER.info(f"Defined optimisation level to {self._optimisation_level}")
        return self

    def logits_as(self, dtype: str) -> "EngineConfigBuilder":
        if dtype not in SUPPORTED_LOGITS_DTYPE:
            raise ValueError(f"logits dtype should be one of {SUPPORTED_LOGITS_DTYPE} (got: {dtype})")

        self._logits_dtype = dtype
        LOGGER.info(f"Defined logits dtype to: {self._logits_dtype}")
        return self

    def with_inference_profile(
        self,
        max_batch_size: int,
        max_prompt_length: int,
        max_new_tokens: int
    ) -> "EngineConfigBuilder":
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size should be >= 1 (got: {max_batch_size})")

        if max_prompt_length < 1:
            raise ValueError(f"max_prompt_length should be >= 1 (got: {max_batch_size})")

        if max_prompt_length >= self._config.max_position_embeddings:
            raise ValueError(
                f"max_prompt_length should be shorter than the maximum length supported by the model."
                f" (got: {max_prompt_length} and"
                f" maximum sequence length supported by the model is {self._config.max_position_embeddings})"
            )

        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens should be >= 1 (got: {max_new_tokens})")

        if max_new_tokens > self._config.max_position_embeddings:
            raise ValueError(
                f"max_new_tokens should be shorter than the maximum length supported by the model."
                f" (got: {max_new_tokens} and"
                f" maximum sequence length supported by the model is {self._config.max_position_embeddings})"
            )

        self._workload_profile = InferenceProfile(max_batch_size, max_prompt_length, max_new_tokens)
        LOGGER.info(f"Defined engine inference profile: {self._workload_profile}")
        return self

    def with_generation_profile(self, num_beams: int) -> "EngineConfigBuilder":
        if num_beams < 1:
            raise ValueError(f"num_beams should be >= 1 (got: {num_beams})")

        self._generation_profile = GenerationProfile(num_beams, -1)
        LOGGER.info(f"Defined engine generation profile: {self._generation_profile}")
        return self

    def with_speculated_decoding(self, max_draft_length: int) -> "EngineConfigBuilder":
        if max_draft_length < 1:
            raise ValueError(f"max_draft_length should be >= 1 (got: {max_draft_length})")

        if self._generation_profile is None:
            raise ValueError(
                "You should specify generation profile first. "
                "Please use EngineConfigBuilder.with_generation_profile()"
            )

        self._generation_profile = GenerationProfile(
            self._generation_profile.num_beams,
            max_draft_length
        )
        LOGGER.info(f"Defined engine generation profile with speculation: {self._generation_profile}")
        return self

    def with_plugins_config(self, plugin_config: PluginConfig) -> "EngineConfigBuilder":
        self._plugin_config = plugin_config
        LOGGER.info(f"Defined plugins config: {plugin_config}")
        return self

    def validate(self) -> bool:
        if self._workload_profile is None:
            raise ValueError("You need to set an inference profile. Use EngineConfigBuilder.with_inference_profile().")

        if self._generation_profile is None:
            raise ValueError("You need to set a generation profile. Use EngineConfigBuilder.with_generation_profile().")

        if self._plugin_config is None:
            raise ValueError("You need to set a plugin profile. Use EngineConfigBuilder.with_plugins_config().")

        max_generated_length = self._workload_profile.max_input_len + self._workload_profile.max_output_len - 1
        if max_generated_length > self._config.max_position_embeddings:
            raise ValueError(
                "max_prompt_length + max_new_tokens should be lesser or equals "
                "to the maximum length supported by the model (got "
                f"max_prompt_length={self._workload_profile.max_input_len}, "
                f"max_new_tokens={self._workload_profile.max_output_len},"
                f"{self._workload_profile.max_input_len + self._workload_profile.max_output_len}"
                f" > {self._config.max_position_embeddings}"
                ")"
            )

        return True
    def build(self) -> EngineConfig:
        self.validate()

        return EngineConfig(
            optimisation_level=self._optimisation_level,
            strongly_typed=self._strongly_typed,
            logits_dtype=self._logits_dtype,
            workload_profile=self._workload_profile,
            generation_profile=self._generation_profile,
            plugins_config=self._plugin_config
        )