import asyncio
import json
import math
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import torch
from tensorrt_llm.bindings.executor import ExecutorConfig, KvCacheConfig
from tensorrt_llm.executor import (
    GenerationExecutor,
    GenerationRequest,
    GenerationResult,
)
from tensorrt_llm.hlapi import SamplingParams
from transformers import GenerationConfig
from optimum.nvidia.hub import HuggingFaceHubModel
from optimum.nvidia.utils.nvml import is_post_ampere


LOGGER = getLogger(__name__)


def read_engine_config_file(path: Path) -> Dict[str, Any]:
    with open(path / "config.json", "r", encoding="utf-8") as config_f:
        return json.load(config_f)


def convert_generation_config(config: "GenerationConfig") -> "SamplingParams":
    return SamplingParams(
        end_id=config.eos_token_id,
        pad_id=config.pad_token_id,
        top_k=config.top_k if config.do_sample else 1,
        top_p=config.top_p,
        temperature=config.temperature,
        beam_width=config.num_beams if config.do_sample else 1,
        bad_token_ids=config.bad_words_ids,
        length_penalty=config.length_penalty,
        repetition_penalty=config.repetition_penalty,
        no_repeat_ngram_size=config.no_repeat_ngram_size
        if config.no_repeat_ngram_size > 0
        else 1,
        min_length=config.min_length if config.min_length > 0 else 1,
        max_new_tokens=config.max_new_tokens,
        return_generation_logits=config.output_logits,
        return_log_probs=not config.renormalize_logits,
    )


def default_executor_config(config: Dict[str, Any]) -> "ExecutorConfig":
    build_config = config["build_config"]
    plugin_config = config["build_config"]["plugin_config"]

    max_blocks_per_sequence = math.floor(
        build_config["max_seq_len"] / plugin_config["tokens_per_block"]
    )

    return ExecutorConfig(
        enable_chunked_context=is_post_ampere(),
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=True,
            max_tokens=build_config["max_beam_width"]
            * plugin_config["tokens_per_block"]
            * max_blocks_per_sequence,
        ),
    )


class InferenceRuntimeBase:
    __slots__ = ("_config", "_executor", "_generation_config", "_sampling_config")

    def __init__(
        self,
        engines_path: Union[str, PathLike],
        generation_config: "GenerationConfig",
        executor_config: Optional["ExecutorConfig"] = None,
        load_engines: bool = True,
    ):
        engines_path = Path(engines_path)

        if not engines_path.exists():
            raise OSError(f"engine folder {engines_path} doesn't exist")

        self._config = read_engine_config_file(engines_path)
        self._generation_config = generation_config
        self._sampling_config = convert_generation_config(generation_config)

        if load_engines:
            self._executor = GenerationExecutor.create(
                engine=engines_path,
                executor_config=executor_config
                or default_executor_config(self._config),
            )

    def generate(
        self,
        inputs: Union[List[int], "torch.IntTensor"],
        generation_config: Optional["GenerationConfig"] = None,
        **kwargs
    ):
        if generation_config is None:
            generation_config = GenerationConfig(**kwargs)

        # Retrieve the sampling config
        sampling = (
            convert_generation_config(generation_config)
            if generation_config
            else self._sampling_config
        )

        if isinstance(inputs, torch.Tensor):
            inputs = inputs.tolist()

        result = self._executor.generate(inputs, sampling_params=sampling)

        # TODO: Fix this
        return result[0].outputs[0].token_ids

    async def agenerate(
        self,
        inputs: Union[List[int], "torch.IntTensor"],
        generation_config: Optional["GenerationConfig"] = None,
        **kwargs
    ) -> List[int]:
        if generation_config is None:
            generation_config = GenerationConfig(**kwargs)

        # Retrieve the sampling config
        sampling = (
            convert_generation_config(generation_config)
            if generation_config
            else self._sampling_config
        )

        if isinstance(inputs, torch.Tensor):
            inputs = inputs.tolist()

        futures = self._executor.generate_async(
            inputs, streaming=False, sampling_params=sampling
        )
        if isinstance(futures, GenerationRequest):
            results = await futures.aresult()
            return results.token_ids
        else:
            results = await asyncio.gather(*[f.aresult() for f in futures])
            return [r.token_ids for r in results]


class CausalLMOutput:
    __slots__ = ("_results",)

    def __init__(
        self, results: Union["GenerationResult", Sequence["GenerationResult"]]
    ):
        self._results = results

    @property
    def logits(self):
        return self._results.token_ids

    @property
    def loss(self) -> None:
        return None


class CausalLM(HuggingFaceHubModel, InferenceRuntimeBase):
    def __init__(
        self,
        engines_path: Union[str, PathLike, Path],
        generation_config: "GenerationConfig",
        executor_config: Optional["ExecutorConfig"] = None,
        load_engines: bool = True,
    ):
        InferenceRuntimeBase.__init__(
            self, engines_path, generation_config, executor_config, load_engines
        )
        HuggingFaceHubModel.__init__(self, engines_path)

    def _save_additional_parcels(self, save_directory: Path):
        self._generation_config.save_pretrained(
            save_directory, "generation_config.json"
        )
