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

import copy
import warnings
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

import tensorrt_llm.bindings as ctrrt
import torch

from transformers import GenerationConfig
from transformers.generation.utils import GenerationMixin


if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from transformers.generation.logits_process import LogitsProcessorList
    from transformers.generation.stopping_criteria import StoppingCriteriaList

LOGGER = getLogger(__name__)

PackedTensor = List[torch.Tensor]


DEFAULT_BATCH_SIZE: int = 1
DEFAULT_PROMPT_LENGTH: int = 128
DEFAULT_BEAM_WIDTH: int = 1


class CompiledModel:
    def __init__(self, engines_folders_path: List[Union[Path, PathLike]]):
        # A compiled model may have several subfolders (e.g. encoder-decoder model).
        self._engines_folders_path = [
            Path(engines_folder_path) for engines_folder_path in engines_folders_path
        ]

    @property
    def engine_path(self) -> Path:
        """
        Return the local path where the engine(s) is/are located
        :return: Path to the folder holding the engine(s) definition(s)
        """
        return self._engines_folders_path


class CausalLM(CompiledModel, GenerationMixin):
    main_input_name = "input_ids"

    __slots__ = (
        "_device",
        "_config",
        "_mapping",
        "_session",
        "_session_config",
        "_use_packed_inputs",
        "max_beam_width",
        "max_batch_size",
        "max_prompt_length",
        "max_output_length",
    )

    def __init__(
        self,
        engines_folders: List[Path],
        *,
        gpus_per_node: int,
        transformers_config: "PretrainedConfig",
        use_cuda_graph: bool = False,
        generation_config: Optional[GenerationConfig] = None,
    ):
        if len(engines_folders) != 1:
            raise ValueError(
                f"For CausalLM, expecting a single engine folder, got: {engines_folders}"
            )
        super().__init__(engines_folders)
        engines_folder = engines_folders[0]

        self._device = torch.device("cuda")
        self._config = ctrrt.GptJsonConfig.parse_file(engines_folder / "config.json")
        self._mapping = ctrrt.WorldConfig.mpi(
            gpus_per_node,
            self._config.tensor_parallelism,
            self._config.pipeline_parallelism,
        )
        self._session_config = ctrrt.GptSessionConfig(
            max_batch_size=self._config.model_config.max_batch_size,
            max_beam_width=self._config.model_config.max_beam_width,
            max_sequence_length=self._config.model_config.max_seq_len,
        )

        self._session_config.cuda_graph_mode = use_cuda_graph

        # Create the engine
        engine_file = self._config.engine_filename(self._mapping)
        self._session = ctrrt.GptSession(
            config=self._session_config,
            model_config=self._config.model_config,
            world_config=self._mapping,
            engine_file=str(engines_folder.joinpath(engine_file)),
        )

        # Additional cached properties
        self._use_packed_inputs = self._config.model_config.use_packed_input
        self.max_batch_size = self._config.model_config.max_batch_size
        self.max_prompt_length = self._config.model_config.max_input_len

        self.max_output_length = self._config.model_config.max_seq_len
        self.max_beam_width = self._session_config.max_beam_width

        if generation_config is None:
            generation_config = GenerationConfig()
        self.generation_config = generation_config

        # Required for GenerationMixin compatibility.
        self.config = transformers_config

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional["LogitsProcessorList"] = None,
        stopping_criteria: Optional["StoppingCriteriaList"] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.LongTensor:

        def raise_unsupported(value: Any, name: str, default: Any = None):
            if value != default:
                raise ValueError(
                    f"{self.__class__.__name__}.generate does not support the argument {name} (got {name}={value}). Please open an issue at https://github.com/huggingface/optimum-nvidia/issues."
                )

        raise_unsupported(stopping_criteria, name="stopping_criteria")
        raise_unsupported(prefix_allowed_tokens_fn, name="prefix_allowed_tokens_fn")
        raise_unsupported(synced_gpus, name="synced_gpus")
        raise_unsupported(logits_processor, name="logits_processor")
        raise_unsupported(assistant_model, name="assistant_model")
        raise_unsupported(streamer, name="streamer")
        raise_unsupported(negative_prompt_ids, name="negative_prompt_ids")
        raise_unsupported(negative_prompt_attention_mask, name="negative_prompt_attention_mask")

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
            # three conditions must be met
            # 1) the generation config must have been created from the model config (`_from_model_config` field);
            # 2) the generation config must have seen no modification since its creation (the hash is the same);
            # 3) the user must have set generation parameters in the model config.
            if (
                self.generation_config._from_model_config
                and self.generation_config._original_object_hash == hash(self.generation_config)
                and self.config._has_non_default_generation_parameters()
            ):
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use and modify the model generation configuration (see"
                        " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                LOGGER.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            LOGGER.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        device = self._device

        seed = model_kwargs.pop("seed", 42)
        # If no GenerationConfig is provided, let's allocate one with default settings
        sampling_config = ctrrt.SamplingConfig(min(generation_config.num_beams, self.max_beam_width))
        sampling_config.random_seed = [seed]
        sampling_config.temperature = [generation_config.temperature]
        sampling_config.top_k = [generation_config.top_k]
        sampling_config.top_p = [generation_config.top_p]
        sampling_config.repetition_penalty = [generation_config.repetition_penalty]
        sampling_config.length_penalty = [generation_config.length_penalty]

        if generation_config.min_new_tokens is not None:
            sampling_config.min_length = [generation_config.min_new_tokens]

        input_ids, _, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )

        with torch.no_grad():
            if not isinstance(input_ids, torch.Tensor):
                raise TypeError("input_ids should be a PyTorch tensor (torch.Tensor)")

            attention_mask = model_kwargs["attention_mask"]
            input_ids, lengths = self._prepare_inputs(input_ids, attention_mask)
            if torch.any(torch.gt(lengths, self.max_prompt_length)):
                raise ValueError(
                    f"Input length {lengths} is bigger than maximum prompt length ({self.max_prompt_length})."
                )

            input_length = input_ids.shape[1]
            trt_inputs = ctrrt.GenerationInput(
                end_id=generation_config.eos_token_id,
                pad_id=generation_config.pad_token_id,
                ids=input_ids.to(device),
                lengths=lengths.to(device),
                packed=self._use_packed_inputs,
            )

            max_new_tokens = generation_config.max_new_tokens
            if max_new_tokens is None or max_new_tokens < 1:
                max_new_tokens = self.max_output_length - input_ids.shape[1]

            trt_inputs.max_new_tokens = max_new_tokens

            # Tensors are being allocated as in/out parameters and TRTLLM will resize
            trt_outputs = ctrrt.GenerationOutput(
                ids=torch.empty(0, device=device, dtype=torch.int32),
                lengths=torch.empty(0, device=device, dtype=torch.int32),
            )

            self._session.generate(trt_outputs, trt_inputs, sampling_config)

            total_length = trt_outputs.lengths
            output_ids = trt_outputs.ids.flatten(0, 1)

            # For some reason not in line with Transformers in case we finish early with BOS token (missing last BOS token).
            if total_length - input_length < max_new_tokens:
                total_length += 1
            
            return output_ids[:, :total_length], total_length

    def _prepare_inputs(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = input_ids.size()
        input_ids = input_ids.int()

        if input_ids.ndim == 1:
            lengths = torch.tensor(shape, dtype=torch.int32, device=self._device)
        elif input_ids.ndim == 2 and shape[0] == 1:
            lengths = torch.tensor([shape[1]], dtype=torch.int32, device=self._device)
        elif attention_mask is not None:
            lengths = attention_mask.sum(dim=1, dtype=torch.int32)
        else:
            warnings.warn(
                "Not enough information to compute the non-padded tensor length. "
                "Please provide an attention_mask to avoid situations where padding"
                " will be attended to in attention modules"
            )

            attention_mask = torch.ones_like(input_ids)
            lengths = torch.tensor(shape, dtype=torch.int32).flatten()

        if self._use_packed_inputs and shape[0] > 1:
            input_ids = torch.masked_select(input_ids, attention_mask.bool()).view(
                1, -1
            )

        return input_ids, lengths


class TensorRTForSpeechSeq2Seq(CompiledModel):
    def __init__(
        self,
        engines_folders: List[Path],
        *,
        gpus_per_node: int,
        transformers_config: "PretrainedConfig",
        use_cuda_graph: bool = False,
        generation_config: Optional[GenerationConfig] = None,
    ):
        super().__init__(engines_folders)
