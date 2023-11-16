import json
import torch
from logging import getLogger
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from huggingface_hub import ModelHubMixin

import tensorrt_llm.bindings as ctrrt


LOGGER = getLogger(__name__)

PackedTensor = List[torch.Tensor]



class TRTEnginePretrainedModel(ModelHubMixin):
    pass


class TRTEngineForCausalLM(TRTEnginePretrainedModel):
    __slots__ = (
        "_config",
        "_mapping",
        "_session",
        "_use_packed_inputs",
        "_max_beam_width",
        "_max_batch_size",
        "_max_prompt_length"
        "_max_new_tokens"
        "_max_output_length"
    )

    def __init__(
        self,
        config: Dict[str, Any],
        engines_folder: Path,
        gpus_per_node: int,
        use_cuda_graph: bool = False
    ):
        super().__init__()

        # TODO avoid the conversion back to str
        self._config = ctrrt.GptJsonConfig.parse(json.dumps(config))
        self._mapping = ctrrt.WorldConfig.mpi(
            gpus_per_node,
            self._config.tensor_parallelism,
            self._config.pipeline_parallelism,
        )
        self._session_config = ctrrt.GptSessionConfig(
            max_batch_size=config["builder_config"].get("max_batch_size", 1),
            max_beam_width=config["builder_config"].get("max_beam_width", 1),
            max_sequence_length=config["builder_config"].get("max_position_embeddings", 512)
        )
        self._session_config.cuda_graph_mode = use_cuda_graph
        # self._session_config.kv_cache_config =

        # Create the engine
        engine_file = self._config.engine_filename(self._mapping)
        self._session = ctrrt.GptSession(
            config=self._session_config,
            model_config=self._config.model_config,
            world_config=self._mapping,
            engine_file=str(engines_folder.joinpath(engine_file))
        )

        # Additional cached properties
        self._use_packed_inputs = config["plugin_config"].get("remove_input_padding", False)
        self._max_batch_size = self._config.model_config.max_batch_size
        self._max_prompt_length = self._config.model_config.max_input_len
        self._max_output_length = self._config.model_config.max_output_len
        self._max_new_tokens = self._max_output_length - self._max_prompt_length
        self._max_beam_width = self._session_config.max_beam_width

    @property
    def config(self) -> ctrrt.GptJsonConfig:
        return self._config

    def generate(self,
        input_ids: Union[PackedTensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 0.0,
        length_penalty: float = 1.0,
        seed: int = 0,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2
    ):
        # If no GenerationConfig is provided, let's allocate one with default settings
        generation_config = ctrrt.SamplingConfig(min(num_beams, self._max_beam_width))
        generation_config.random_seed = [seed]
        generation_config.temperature = [temperature]
        generation_config.top_k = [top_k]
        generation_config.top_p = [top_p]
        generation_config.repetition_penalty = [repetition_penalty]
        generation_config.length_penalty = [length_penalty]

        with torch.no_grad():
            if isinstance(input_ids, torch.Tensor):

                if attention_mask is not None:
                    lengths = attention_mask.sum(dim=1, dtype=torch.int32)
                elif input_ids.ndim == 1:
                    lengths = torch.tensor([input_ids.size(0)], dtype=torch.int32)
                elif input_ids.ndim == 2 and input_ids.size(0) == 1:
                    lengths = torch.tensor([input_ids.size(1)], dtype=torch.int32)
                else:
                    raise NotImplementedError("Cannot compute lengths from torch.Tensor without attention_mask")
            else:
                input_ids = torch.nested.nested_tensor(input_ids)
                lengths = torch.tensor([tensor.size(0) for tensor in input_ids], dtype=torch.int32)

            if torch.any(torch.gt(lengths, self._max_prompt_length)):
                raise ValueError(f"Input length is bigger than maximum prompt length ({self._max_prompt_length}).")

            trt_inputs = ctrrt.GenerationInput(
                end_id=eos_token_id,
                pad_id=pad_token_id,
                ids=input_ids.view((input_ids.size(0), -1)).int(),
                lengths=lengths,
                packed=self._use_packed_inputs
            )

            # Define some additional parameters based on the above
            if max_new_tokens > self._max_new_tokens:
                LOGGER.warning(f"max_new_tokens {max_new_tokens} reduced to {self._max_new_tokens} to match engine.")

            trt_inputs.max_new_tokens = min(max_new_tokens, self._max_new_tokens)

            trt_outputs = ctrrt.GenerationOutput(
                ids=torch.empty((self._max_batch_size, self._max_output_length), dtype=torch.int32),
                lengths=torch.empty(self._max_batch_size, dtype=torch.int32)
            )
            self._session.generate(trt_outputs, trt_inputs, generation_config)

            return trt_outputs.ids

