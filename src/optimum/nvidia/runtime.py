import json
import warnings
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import tensorrt_llm.bindings as ctrrt
import torch
from huggingface_hub import ModelHubMixin
from transformers import AutoTokenizer

from .utils.constants import DEFAULT_ENGINE_FOLDER, OPTIMUM_NVIDIA_CONFIG_FILE
from .builder import TensorRTEngineBuilder
from optimum.nvidia.configs import TransformersConfig
from optimum.nvidia.models import ConvertibleModel
from optimum.nvidia.utils import get_local_empty_folder, get_user_agent


LOGGER = getLogger(__name__)

PackedTensor = List[torch.Tensor]


DEFAULT_BATCH_SIZE: int = 1
DEFAULT_PROMPT_LENGTH: int = 128
DEFAULT_BEAM_WIDTH: int = 1


class TensorRTPreTrainedModel(ModelHubMixin):
    def __init__(self, engines_folder_path: Union[Path, PathLike]):
        self._engines_folder_path = Path(engines_folder_path)

    @property
    def engine_path(self) -> Path:
        """
        Return the local path where the engine(s) is/are located
        :return: Path to the folder holding the engine(s) definition(s)
        """
        return self._engines_folder_path

    def _save_pretrained(self, save_directory: Path) -> None:
        # All engines are serialized on the disk, let's first check if save_directory is not
        # just the path where the engines were serialized.
        # In this case it would just be a no-op

        if save_directory != self._engines_folder_path:
            if not any(save_directory.iterdir()):
                from shutil import copytree

                copytree(self._engines_folder_path, save_directory, dirs_exist_ok=True)
            else:
                raise ValueError(f"{save_directory} is not empty")

    @classmethod
    def _from_pretrained(
        cls: Type[ConvertibleModel],
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
    ) -> ConvertibleModel:
        # Build config
        optimization_level = model_kwargs.get("opt_level", 2)
        tp_degree = model_kwargs.get("tp", 1)
        pp_degree = model_kwargs.get("pp", 1)
        gpus_per_node = model_kwargs.get("gpus_per_node", 1)
        world_size = model_kwargs.get("world_size", gpus_per_node)
        use_cuda_graph = model_kwargs.get("use_cuda_graph", False)

        # Let's make sure we have the config
        model_config = model_kwargs.get("config", None)
        if not model_config:
            raise ValueError(
                "Original model configuration (config.json) was not found."
                "The model configuration is required to build TensorRT-LLM engines."
            )

        model_config = TransformersConfig(model_config)

        model_id_or_path = Path(model_id)
        if model_id_or_path.exists() and model_id_or_path.is_dir():
            LOGGER.debug(f"Loading prebuild engine(s) from: {model_id_or_path}")
            engine_folder = model_id_or_path

        else:
            LOGGER.debug(f"Building engine(s) from model's hub: {model_id_or_path}")
            builder = model_kwargs.get("builder", None)

            # If the builder is not provided, let's create a new one
            if not builder:
                LOGGER.debug("No builder provided, using default one")

                raise NotImplementedError("This should be more adaptable")

                # Define some parameters the user can provide
                model_dtype = model_kwargs.get("dtype", "float16")
                use_fp8 = model_kwargs.get("use_fp8", False)
                max_batch_size = model_kwargs.get("max_batch_size", DEFAULT_BATCH_SIZE)
                max_prompt_length = model_kwargs.get("max_prompt_length", DEFAULT_PROMPT_LENGTH)
                max_new_tokens = model_kwargs.get("max_new_tokens", -1)
                max_beam_width = model_kwargs.get("max_beam_width", DEFAULT_BEAM_WIDTH)

                # max new tokens can be determined from the maximum sequence length supported by the model - len(prompt)
                if max_new_tokens < 1:
                    max_new_tokens = max(max_new_tokens, model_config.max_sequence_length - max_prompt_length)

                builder = (
                    TensorRTEngineBuilder(model_id, model_config, cls.ADAPTER)
                    .to(model_dtype)
                    .shard(tp_degree, pp_degree, world_size, gpus_per_node)
                    .with_generation_profile(max_batch_size, max_prompt_length, max_new_tokens)
                    .with_sampling_strategy(max_beam_width)
                )

                if use_fp8:
                    from tensorrt_llm.quantization import QuantMode

                    from optimum.nvidia.configs import QuantizationConfig
                    from optimum.nvidia.quantization import get_default_calibration_dataset

                    num_calibration_samples = model_kwargs.get("num_calibration_samples", 512)
                    calibration = get_default_calibration_dataset(num_calibration_samples)

                    LOGGER.debug(f"Calibrating for float8 (num_calibration_samples={num_calibration_samples}).")

                    if hasattr(calibration, "tokenize"):
                        tokenizer = AutoTokenizer.from_pretrained(
                            model_id, use_agent=get_user_agent(), padding_side="left"
                        )

                        # Let's make sure the calibration see some padding
                        # TODO: Do we need this? We use the remove_input_padding plugins most of the time ...
                        # if not tokenizer.pad_token and tokenizer.eos_token:
                        #     tokenizer.pad_token = tokenizer.eos_token
                        #     pad_to_multiple_of = 8
                        # else:
                        #     pad_to_multiple_of = None

                        calibration.tokenize(
                            tokenizer, max_length=max_prompt_length + max_new_tokens, pad_to_multiple_of=1
                        )

                    builder.with_quantization_profile(
                        QuantizationConfig(QuantMode.from_description(use_fp8_qdq=True, use_fp8_kv_cache=True)),
                        calibration,
                    )

            # Retrieve the path where to store and use this to store the TensorRTEngineBuilder artifacts
            engine_folder = get_local_empty_folder(DEFAULT_ENGINE_FOLDER)
            builder.build(engine_folder, optimization_level)

        # Let's load the TensorRT engine config as a JSON file
        with open(engine_folder.joinpath(OPTIMUM_NVIDIA_CONFIG_FILE), "r") as trt_config_f:
            trt_config = json.load(trt_config_f)

        return cls(trt_config, engine_folder, gpus_per_node, use_cuda_graph)


class TensorRTForCausalLM(TensorRTPreTrainedModel):
    __slots__ = (
        "_device",
        "_config",
        "_mapping",
        "_session",
        "_use_packed_inputs",
        "max_beam_width",
        "max_batch_size",
        "max_prompt_length",
        "max_output_length",
    )

    def __init__(self, config: Dict[str, Any], engines_folder: Path, gpus_per_node: int, use_cuda_graph: bool = False):
        super().__init__(engines_folder)

        self._device = torch.device("cuda")
        self._config = ctrrt.GptJsonConfig.parse(json.dumps(config))
        self._mapping = ctrrt.WorldConfig.mpi(
            gpus_per_node,
            self._config.tensor_parallelism,
            self._config.pipeline_parallelism,
        )
        self._session_config = ctrrt.GptSessionConfig(
            max_batch_size=config["builder_config"].get("max_batch_size", 1),
            max_beam_width=config["builder_config"].get("max_beam_width", 1),
            max_sequence_length=config["builder_config"]["max_output_len"],
        )
        self._session_config.cuda_graph_mode = use_cuda_graph
        # self._session_config.kv_cache_config =

        # Create the engine
        engine_file = self._config.engine_filename(self._mapping)
        self._session = ctrrt.GptSession(
            config=self._session_config,
            model_config=self._config.model_config,
            world_config=self._mapping,
            engine_file=str(engines_folder.joinpath(engine_file)),
        )

        # Additional cached properties
        self._use_packed_inputs = config["plugin_config"].get("remove_input_padding", False)
        self.max_batch_size = self._config.model_config.max_batch_size
        self.max_prompt_length = self._config.model_config.max_input_len
        self.max_output_length = self._config.model_config.max_output_len
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
        repetition_penalty: float = 0.0,
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
            input_ids = torch.masked_select(input_ids, attention_mask.bool()).view(1, -1)

        return input_ids, lengths

class TensorRTForSpeechSeq2Seq(TensorRTPreTrainedModel):
    # TODO: implement
    pass
