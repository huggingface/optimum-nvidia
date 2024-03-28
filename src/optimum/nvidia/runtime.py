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

import warnings
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import List, Optional, Tuple, Union

import tensorrt_llm.bindings as ctrrt
import torch


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


class CausalLM(CompiledModel):
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
        use_cuda_graph: bool = False,
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

    @property
    def config(self) -> ctrrt.GptJsonConfig:
        return self._config

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = -1,
        min_length: int = -1,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        seed: int = 0,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self._device

        # If no GenerationConfig is provided, let's allocate one with default settings
        generation_config = ctrrt.SamplingConfig(min(num_beams, self.max_beam_width))
        generation_config.random_seed = [seed]
        generation_config.temperature = [temperature]
        generation_config.top_k = [top_k]
        generation_config.top_p = [top_p]
        generation_config.repetition_penalty = [repetition_penalty]
        generation_config.length_penalty = [length_penalty]

        if min_length > 0:
            generation_config.min_length = [min_length]

        with torch.no_grad():
            if not isinstance(input_ids, torch.Tensor):
                raise TypeError("input_ids should be a PyTorch tensor (torch.Tensor)")

            input_ids, lengths = self._prepare_inputs(input_ids, attention_mask)
            if torch.any(torch.gt(lengths, self.max_prompt_length)):
                raise ValueError(
                    f"Input length {lengths} is bigger than maximum prompt length ({self.max_prompt_length})."
                )

            trt_inputs = ctrrt.GenerationInput(
                end_id=eos_token_id,
                pad_id=pad_token_id,
                ids=input_ids.to(device),
                lengths=lengths.to(device),
                packed=self._use_packed_inputs,
            )

            if max_new_tokens is None or max_new_tokens < 1:
                max_new_tokens = self.max_output_length - input_ids.shape[1]

            trt_inputs.max_new_tokens = max_new_tokens

            # Tensors are being allocated as in/out parameters and TRTLLM will resize
            trt_outputs = ctrrt.GenerationOutput(
                ids=torch.empty(0, device=device, dtype=torch.int32),
                lengths=torch.empty(0, device=device, dtype=torch.int32),
            )

            self._session.generate(trt_outputs, trt_inputs, generation_config)

            return trt_outputs.ids, trt_outputs.lengths

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
        use_cuda_graph: bool = False,
    ):
        super().__init__(engines_folders)

        # TODO: implement this.
