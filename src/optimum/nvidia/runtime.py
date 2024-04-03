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
import warnings
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import List, Optional, Tuple, Union, Callable, Any, Dict, Iterator
from tensorrt_llm import str_dtype_to_torch
from collections import OrderedDict
import tensorrt_llm.bindings as ctrrt
import torch
from typing import TYPE_CHECKING
from tensorrt_llm.runtime.session import TensorInfo
from tensorrt_llm.runtime import SamplingConfig
from tensorrt_llm._utils import str_dtype_to_trt, trt_dtype_to_torch
from optimum.nvidia.config import dtype_to_str
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE, TASK_IDS
from transformers import GenerationConfig

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from transformers.generation.logits_process import LogitsProcessorList
    from transformers.generation.stopping_criteria import StoppingCriteriaList
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.streamers import BaseStreamer

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

        self.transformers_config = transformers_config

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
        transformers_config: "PretrainedConfig",
        use_cuda_graph: bool = False,
        generation_config: Optional[GenerationConfig] = None,
    ):
        super().__init__(engines_folders)
        
    def encoder(self,
        input_features: torch.Tensor,
    ):
        if dtype_to_str(input_features.dtype) != self.dtype:
            LOGGER.warning(f"input_features should be of dtype {self.dtype}, got {dtype_to_str(input_features.dtype)}. Automatically casting to {self.dtype}.")
            input_features = input_features.to(str_dtype_to_torch(self.dtype))

        input_lengths = torch.tensor(
            [input_features.shape[2] // 2 for _ in range(input_features.shape[0])],
            dtype=torch.int32,
            device=input_features.device)

        inputs = OrderedDict()
        inputs['x'] = input_features
        inputs['input_lengths'] = input_lengths

        output_list = [
            TensorInfo('x', str_dtype_to_trt(self.dtype), input_features.shape),
            TensorInfo('input_lengths', str_dtype_to_trt('int32'),
                       input_lengths.shape)
        ]

        output_info = (self.encoder_session).infer_shapes(output_list)

        LOGGER.debug(f'output info {output_info}')
        outputs = {
            t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device='cuda')
            for t in output_info
        }
        
        stream = torch.cuda.current_stream()

        ok = self.encoder_session.run(
            inputs=inputs,
            outputs=outputs,
            stream=stream.cuda_stream
        )

        assert ok, 'Engine execution failed'
        stream.synchronize()

        return outputs['output']

    def _retrieve_init_tokens(self, input_features, generation_config, config, num_segment_frames, kwargs):
        def replace_or_add(lst: List[int], num: int, itr: Iterator[int]):
            """short function to replace num with a itr in lst"""
            found = any(i in lst for i in itr)
            if found:
                lst = [num if i in itr else i for i in lst]
            else:
                lst.append(num)
            return lst

        task = getattr(generation_config, "task", None)
        language = getattr(generation_config, "language", None)

        if kwargs.get("forced_decoder_ids", None) is not None:
            forced_decoder_ids = kwargs["forced_decoder_ids"]
        elif hasattr(generation_config, "forced_decoder_ids") and generation_config.forced_decoder_ids is not None:
            forced_decoder_ids = generation_config.forced_decoder_ids

            if language is None and task is None and forced_decoder_ids[0][1] is None:
                LOGGER.warning_once(
                    "Due to a bug fix in https://github.com/huggingface/transformers/pull/28687 transcription using a multilingual Whisper will default to language detection followed by transcription instead of translation to English."
                    "This might be a breaking change for your use case. If you want to instead always translate your audio to English, make sure to pass `language='en'`."
                )
        elif hasattr(config, "forced_decoder_ids") and config.forced_decoder_ids is not None:
            forced_decoder_ids = config.forced_decoder_ids
        else:
            forced_decoder_ids = None

        if forced_decoder_ids is not None and task is not None:
            LOGGER.info(
                f"You have passed task={task}, but also have set `forced_decoder_ids` to {forced_decoder_ids} which creates a conflict. `forced_decoder_ids` will be ignored in favor of task={task}."
            )
            forced_decoder_ids = None
        elif forced_decoder_ids is not None and language is not None:
            LOGGER.info(
                f"You have passed language={language}, but also have set `forced_decoder_ids` to {forced_decoder_ids} which creates a conflict. `forced_decoder_ids` will be ignored in favor of language={language}."
            )
            forced_decoder_ids = None

        init_tokens = [generation_config.decoder_start_token_id]
        if forced_decoder_ids is not None and forced_decoder_ids[0][0] == 1:
            i = 1
            while len(forced_decoder_ids) > 0 and forced_decoder_ids[0][0] == i:
                init_tokens += [forced_decoder_ids[0][1]]
                forced_decoder_ids = forced_decoder_ids[1:]
                i += 1

            if len(forced_decoder_ids) > 0:
                raise ValueError(
                    f"You are using token ids in `forced_decoder_ids` that do not seem to correctly follow the prompt pattern of Whisper. Make sure that {forced_decoder_ids} has an entry for all indices >= 1 and < {forced_decoder_ids[0][0]}.",
                )

        # from v4.39 the forced decoder ids are always None in favour of decoder input ids
        generation_config.forced_decoder_ids = None

        is_lang_id_undefined = len(init_tokens) <= 1 or (len(init_tokens) > 1 and init_tokens[1] is None)
        if language is not None:
            if language in generation_config.lang_to_id.keys():
                language_token = language
            elif language in TO_LANGUAGE_CODE.keys():
                language_token = f"<|{TO_LANGUAGE_CODE[language]}|>"
            elif language in TO_LANGUAGE_CODE.values():
                language_token = f"<|{language}|>"
            else:
                is_language_code = len(language) == 2
                raise ValueError(
                    f"Unsupported language: {language}. Language should be one of:"
                    f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
                )
            if language_token not in generation_config.lang_to_id:
                raise ValueError(
                    f"{language_token} is not supported by this specific model as it is not in the `generation_config.lang_to_id`."
                    "(You should just add it to the generation config)"
                )

            lang_id = generation_config.lang_to_id[language_token]

            # if language is defined it'll overwrite language ids that might have already been defined via the generation_config
            replace_or_add(init_tokens, lang_id, generation_config.lang_to_id.values())
        elif hasattr(generation_config, "lang_to_id") and is_lang_id_undefined:
            # language is not defined or intentially set to `None` to trigger language detection
            lang_ids = self.detect_language(
                input_features=input_features,
                encoder_outputs=kwargs.get("encoder_outputs", None),
                generation_config=generation_config,
                num_segment_frames=num_segment_frames,
            )

            if torch.unique(lang_ids).shape[0] > 1:
                raise ValueError(
                    "Multiple languages detected when trying to predict the most likely target language for transcription. It is currently not supported to transcribe to different languages in a single batch. Please make sure to either force a single language by passing `language='...'` or make sure all input audio is of the same language."
                )

            lang_id = lang_ids[0].item()

            # append or replace lang_id to init_tokens
            if len(init_tokens) > 1:
                init_tokens[1] = lang_id
            else:
                init_tokens.append(lang_id)

        if task is not None:
            if task in TASK_IDS:
                init_tokens.append(generation_config.task_to_id[generation_config.task])
                task_id = generation_config.task_to_id[generation_config.task]

                # if task is defined it'll overwrite task ids that might have already been defined via the generation_config
                replace_or_add(init_tokens, task_id, generation_config.task_to_id.values())
            else:
                raise ValueError(f"The `{task}`task is not supported. The task should be one of `{TASK_IDS}`")
        elif language is not None and hasattr(generation_config, "task_to_id"):
            # if language is defined, but no task id is in `init_tokens`, default to transcribe
            if not any(i in init_tokens for i in generation_config.task_to_id.values()):
                init_tokens.append(generation_config.task_to_id["transcribe"])

        if (
            not generation_config.return_timestamps
            and hasattr(generation_config, "no_timestamps_token_id")
            and init_tokens[-1] != generation_config.no_timestamps_token_id
        ):
            init_tokens.append(generation_config.no_timestamps_token_id)
        elif generation_config.return_timestamps and init_tokens[-1] == generation_config.no_timestamps_token_id:
            LOGGER.info(
                "<|notimestamps|> prompt token is removed from generation_config since `return_timestamps` is set to `'True'`."
            )
            init_tokens = init_tokens[:-1]

        # let's make sure we don't pass `None` tokens as prompt tokens
        init_tokens = [t for t in init_tokens if t is not None]

        return init_tokens

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional["GenerationConfig"] = None,
        logits_processor: Optional["LogitsProcessorList"] = None,
        stopping_criteria: Optional["StoppingCriteriaList"] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if inputs.device.type != "cuda":
            raise ValueError(f"TensorRT-LLM only supports inputs on CUDA device. Got: inputs.device = {inputs.device}")
    
        def raise_unsupported(value: Any, name: str):
            if value is not None:
                raise ValueError(f"TensorRTForSpeechSeq2Seq.generate does not support {name} (got {value}). Please open an issue at https://github.com/huggingface/optimum-nvidia/issues.")
        
        raise_unsupported(logits_processor, name="logits_processor")
        raise_unsupported(stopping_criteria, name="stopping_criteria")
        raise_unsupported(prefix_allowed_tokens_fn, name="prefix_allowed_tokens_fn")
        raise_unsupported(synced_gpus, name="synced_gpus")
        raise_unsupported(assistant_model, name="assistant_model")
        raise_unsupported(streamer, name="streamer")
        raise_unsupported(negative_prompt_ids, name="negative_prompt_ids")
        raise_unsupported(negative_prompt_attention_mask, name="negative_prompt_attention_mask")

        if generation_config is None:
            generation_config = self.generation_config
        
        num_beams = kwargs.pop("num_beams", generation_config.num_beams)
        
        encoder_outputs = self.encoder(inputs)

        batch_size = inputs.shape[0]

        input_stride = 1 * 2  # encoder's conv1 stride * encoder's conv2 stride
        num_segment_frames = input_stride * self.transformers_config.max_source_positions
        init_tokens = self._retrieve_init_tokens(
            inputs,
            generation_config=generation_config,
            config=self.transformers_config,
            num_segment_frames=num_segment_frames,
            kwargs=kwargs,
        )
        one_tensor = torch.ones((batch_size, 1), device="cuda", dtype=torch.long)
        decoder_input_ids = torch.cat([t * one_tensor for t in init_tokens], dim=-1)
        
        max_new_tokens = kwargs.pop("max_new_tokens", generation_config.max_new_tokens)
        if max_new_tokens is None:
            # Transformers' GenerationConfig.max_new_tokens defaults to None.
            if generation_config.max_length is not None:
                max_new_tokens = generation_config.max_length - decoder_input_ids.shape[1]
            else:
                raise ValueError("Please specifiy the argument `max_new_tokens`.")

        if max_new_tokens + decoder_input_ids.shape[-1] > self.transformers_config.max_target_positions:
            max_new_tokens = kwargs.get("max_new_tokens", 0)
            raise ValueError(
                f"The length of `decoder_input_ids` equal `prompt_ids` plus special start tokens is {decoder_input_ids.shape[-1]}, and the `max_new_tokens` "
                f"is {max_new_tokens}. Thus, the combined length of "
                f"`decoder_input_ids` and `max_new_tokens` is: {max_new_tokens + decoder_input_ids.shape[-1]}. This exceeds the "
                f"`max_target_positions` of the Whisper model: {self.transformers_config.max_target_positions}. "
                "You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, "
                f"so that their combined length is less than {self.transformers_config.max_target_positions}."
            )

        encoder_input_lengths = torch.tensor(
            [encoder_outputs.shape[1] for x in range(encoder_outputs.shape[0])],
            dtype=torch.int32,
            device=inputs.device)

        decoder_input_lengths = torch.tensor([decoder_input_ids.shape[-1] for _ in range(decoder_input_ids.shape[0])], dtype=torch.int32, device='cuda')
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        cross_attention_mask = torch.ones([encoder_outputs.shape[0], 1, encoder_outputs.shape[1]], device=inputs.device, dtype=torch.int32)

        sampling_config = SamplingConfig(
            end_id=generation_config.eos_token_id,
            pad_id=generation_config.pad_token_id,
            num_beams=num_beams
        )

        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            beam_width=num_beams,
            encoder_max_input_length=encoder_outputs.shape[1])

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()

        # output_ids of shape [batch_size, beam_width, output_len]
        output_ids = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_input_lengths,
            cross_attention_mask=cross_attention_mask,
        )
        torch.cuda.synchronize()

        return output_ids[:, 0, :torch.max(self.decoder_generation_session.sequence_length_buffer) + 1]
