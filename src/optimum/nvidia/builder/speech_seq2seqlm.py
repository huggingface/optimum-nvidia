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


from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import Dict, Optional, Type, Union

from huggingface_hub import ModelHubMixin
from huggingface_hub.hub_mixin import T  # What is this? T?
from tensorrt_llm import Mapping as Shard
from tensorrt_llm._utils import trt_version
from tensorrt_llm.builder import Builder
from tensorrt_llm.quantization import QuantMode
from transformers import AutoModelForSpeechSeq2Seq

from ..configs import ModelConfig, QuantizationConfig, TransformersConfig
from ..lang import DataType
from ..models.whisper import WhisperDecoderWeightAdapter, WhisperEncoderWeightAdapter
from .base import TensorRTEngineBuilder


LOGGER = getLogger(__name__)


# TODO: In the future, clean up the `config.config["xxx"]` by using Optimum's NormalizedConfig.
class TensorRTWhisperEncoderEngineBuilder(TensorRTEngineBuilder):
    LOADING_CLASS = AutoModelForSpeechSeq2Seq

    def prepare_inputs(self, model):
        inputs = model.prepare_inputs(
            max_batch_size=self._optimization_profile.max_batch_size,
        )

        # TRT-LLM's prepare_inputs should always return a tuple, but it was not the case for WhisperEncoder.
        # The patch https://github.com/NVIDIA/TensorRT-LLM/commit/c89653021e66ca78c55f02b366f404455bc12e8d
        # is not yet included in huggingface/optimum-nvidia:latest.
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        return inputs

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
        for prop, (min_value, max_value) in [
            ("max_batch_size", (1, None)),
        ]:
            prop_value = getattr(optim_profile, prop)
            if prop_value < min_value:
                raise ValueError(f"Invalid value ({prop_value}) for {prop}. Needs to be >= {min_value}")

            if max_value is not None and prop_value > max_value:
                raise ValueError(f"Invalid value ({prop_value}) for {prop}. Needs to be <= {max_value}")

        return True

    def create_builder_config(
        self, tensorrt_llm_builder: Builder, shard: Shard, is_parallel: bool, opt_level: Optional[int]
    ):
        """
        Prepares the builder for the model. This is kept for backward compatibility in the base class for Llama, but this should be overridden for each architecture as `Builder.create_builder_config` takes different arguments depending on the architecture.
        """
        config = self._model_config
        qconfig = self._quantization_config

        is_multilingual = config.vocab_size >= 51865
        num_languages = config.vocab_size - 51765 - int(is_multilingual)

        if opt_level is not None:
            # TensorRT-LLM example always uses opt_level=None.
            LOGGER.warning(f"Ignoring opt_level={opt_level} for Whisper encoder.")

        if shard.tp_size != 1:
            # TensorRT-LLM/examples/whisper/run.py hard-codes TP to 1 for Whisper encoder.
            LOGGER.warning(f"Ignoring shard.tp_size={shard.tp_size} for Whisper encoder, using tensor_parallel=1.")

        build_config = tensorrt_llm_builder.create_builder_config(
            name=config["model_type"],
            precision=self._dtype.value,
            fp8=qconfig.mode.has_fp8_qdq(),
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            max_batch_size=self._optimization_profile.max_batch_size,
            tensor_parallel=1,
            use_refit=False,
            quant_mode=self._quantization_config.mode,
            huggingface=dict(**config),
            tensorrt=trt_version(),
            n_mels=config.config["num_mel_bins"],
            num_languages=num_languages,
            num_heads=config.config["encoder_attention_heads"],
        )

        return build_config


class TensorRTWhisperDecoderEngineBuilder(TensorRTEngineBuilder):
    LOADING_CLASS = AutoModelForSpeechSeq2Seq

    def prepare_inputs(self, model):
        inputs = model.prepare_inputs(
            max_batch_size=self._optimization_profile.max_batch_size,
            max_beam_width=self._beam_width,
            max_decoder_input_len=self._optimization_profile.max_prompt_length,
            max_new_tokens=self._optimization_profile.max_new_tokens,
            max_encoder_input_len=self._model_config.config["max_source_positions"],
        )

        return inputs

    def create_builder_config(
        self, tensorrt_llm_builder: Builder, shard: Shard, is_parallel: bool, opt_level: Optional[int]
    ):
        config = self._model_config
        qconfig = self._quantization_config

        if opt_level is not None:
            # TensorRT-LLM example always uses opt_level=None.
            LOGGER.warning(f"Ignoring opt_level={opt_level} for Whisper decoder.")

        build_config = tensorrt_llm_builder.create_builder_config(
            name=config["model_type"],
            precision=self._dtype.value,
            fp8=qconfig.mode.has_fp8_qdq(),
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            max_batch_size=self._optimization_profile.max_batch_size,
            tensor_parallel=shard.tp_size,
            use_refit=False,
            quant_mode=self._quantization_config.mode,
            huggingface=dict(**config),
            tensorrt=trt_version(),
            hidden_act="gelu",
            max_position_embeddings=config.config["max_target_positions"],
            apply_query_key_layer_scaling=False,
            max_input_len=self._optimization_profile.max_prompt_length,
            max_output_len=self._optimization_profile.max_output_length,
            cross_attention=True,
            has_position_embedding=True,
            has_token_type_embedding=False,
            num_heads=config.config["decoder_attention_heads"],
        )

        return build_config


class TensorRTForSpeechSeq2SeqEngineBuilder(ModelHubMixin):
    def __init__(self, model_id_or_path: Union[str, "PathLike"], config: "ModelConfig"):
        # Model
        self._model_id_or_path: Union[str, "PathLike"] = model_id_or_path
        self._model_config: "ModelConfig" = config

        self.encoder_builder = TensorRTWhisperEncoderEngineBuilder(
            model_id_or_path, config, WhisperEncoderWeightAdapter
        )
        self.decoder_builder = TensorRTWhisperDecoderEngineBuilder(
            model_id_or_path, config, WhisperDecoderWeightAdapter
        )

    @classmethod
    def _from_pretrained(
        cls: Type["T"],
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, "Path"]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Optional[Union[str, bool]],
        **model_kwargs,
    ) -> "T":
        config = model_kwargs.get("config", None)

        if config and not isinstance(config, TransformersConfig):
            config = TransformersConfig(config)
        else:
            raise ValueError(f"Unsupported configuration type ({type(config).__name__})")

        return cls(model_id, config)

    def build(self, output_path: "PathLike", optimization_level: int = None) -> "PathLike":
        LOGGER.info("Building TensorRT encoder engine...")
        self.encoder_builder.build(Path(output_path, "encoder"), optimization_level)

        LOGGER.info("Building TensorRT decoder engine...")
        self.decoder_builder.build(Path(output_path, "decoder"), optimization_level)

    def to(self, dtype: Union[str, "DataType"]) -> "TensorRTForSpeechSeq2SeqEngineBuilder":
        self.encoder_builder.to(dtype)
        self.decoder_builder.to(dtype)

        return self

    def shard(
        self, tp_degree: int, pp_degree: int, world_size: int, num_gpus_per_node: int
    ) -> "TensorRTForSpeechSeq2SeqEngineBuilder":
        self.encoder_builder.shard(
            tp_degree=tp_degree, pp_degree=pp_degree, world_size=world_size, num_gpus_per_node=num_gpus_per_node
        )
        self.decoder_builder.shard(
            tp_degree=tp_degree, pp_degree=pp_degree, world_size=world_size, num_gpus_per_node=num_gpus_per_node
        )

        return self

    def with_generation_profile(
        self,
        max_batch_size: int,
        max_prompt_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        max_output_length: Optional[int] = None,
    ) -> "TensorRTForSpeechSeq2SeqEngineBuilder":
        if max_prompt_length is not None or max_new_tokens is not None or max_output_length is not None:
            LOGGER.warning(
                f"The builder with_generation_profile arguments max_prompt_length={max_prompt_length}, max_new_tokens={max_new_tokens}, max_output_length={max_output_length} are ignored for Whisper. Using: max_prompt_length=1, max_new_tokens={self._model_config.max_sequence_length}, max_output_length={self._model_config.max_sequence_length}."
            )

        max_prompt_length = 1
        max_new_tokens = self._model_config.max_sequence_length - 1
        max_output_length = self._model_config.max_sequence_length

        self.encoder_builder.with_generation_profile(
            max_batch_size=max_batch_size,
        )

        self.decoder_builder.with_generation_profile(
            max_batch_size=max_batch_size,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            max_output_length=max_output_length,
        )

        return self

    def with_sampling_strategy(self, num_beams: int):
        self.decoder_builder.with_sampling_strategy(num_beams)

        return self
