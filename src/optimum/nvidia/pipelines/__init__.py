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

from os import PathLike
from typing import Dict, Optional, Tuple, Type, Union

from huggingface_hub import model_info
from tensorrt_llm import Module
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from optimum.nvidia import AutoModelForCausalLM
from optimum.nvidia.pipelines.text_generation import TextGenerationPipeline

from .base import Pipeline


SUPPORTED_MODEL_WITH_TASKS: Dict[str, Dict[str, Tuple[Type[Pipeline], Type]]] = {
    "gemma": {"text-generation": (TextGenerationPipeline, AutoModelForCausalLM)},
    "llama": {"text-generation": (TextGenerationPipeline, AutoModelForCausalLM)},
    "mistral": {"text-generation": (TextGenerationPipeline, AutoModelForCausalLM)},
    "mixtral": {"text-generation": (TextGenerationPipeline, AutoModelForCausalLM)},
}


def get_target_class_for_model_and_task(task: str, architecture: str) -> Optional[Type]:
    task_ = SUPPORTED_MODEL_WITH_TASKS.get(task, None)
    if not task_:
        raise NotImplementedError(f"Task {task} is not supported yet.")

    target = task_.get(architecture, None)

    if not target:
        raise NotImplementedError(
            f"Architecture {architecture} is not supported for task {task}. "
            f"Only the following architectures are: {list(task_.keys())}"
        )

    return target


def pipeline(
    task: str = None,
    model: Union[str, PathLike, Module] = None,
    tokenizer: Optional[
        Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast]
    ] = None,
    **kwargs,
):
    """
    Utility factory method to build a [`Pipeline`].

    Pipelines are made of:

        - A [tokenizer](tokenizer) in charge of mapping raw textual input to token.
        - A [model](model) to make predictions from the inputs.
        - Some (optional) post processing for enhancing model's output.

    Args:
        task (`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:
                - `"text-generation"`: will return a [`TextGenerationPipeline`]:.
        model (`str` or [`PreTrainedModel`] or [`TFPreTrainedModel`], *optional*):
            The model that will be used by the pipeline to make predictions. This can be a model identifier or an
            actual instance of a pretrained model inheriting from [`PreTrainedModel`] (for PyTorch) or
            [`TFPreTrainedModel`] (for TensorFlow).

            If not provided, the default for the `task` will be loaded.
        tokenizer (`str` or [`PreTrainedTokenizer`], *optional*):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained tokenizer inheriting from [`PreTrainedTokenizer`].

            If not provided, the default tokenizer for the given `model` will be loaded (if it is a string). If `model`
            is not specified or not a string, then the default tokenizer for `config` is loaded (if it is a string).
            However, if `config` is also not given or not a string, then the default tokenizer for the given `task`
            will be loaded.

    """

    try:
        info = model_info(model)
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate the pipeline inferring the task for model {model}: {e}"
        )

    # Retrieve the model type
    model_type = info.config.get("model_type", None)
    if not model_type:
        raise RuntimeError(f"Failed to infer model type for model {model}")
    elif model_type not in SUPPORTED_MODEL_WITH_TASKS:
        raise NotImplementedError(f"Model type {model_type} is not currently supported")

    if not task and getattr(info, "library_name", "transformers") == "transformers":
        if not info.pipeline_tag:
            raise RuntimeError(
                f"Failed to infer the task for model {model}, please use `task` parameter"
            )
        task = info.pipeline_tag

    if task not in SUPPORTED_MODEL_WITH_TASKS[model_type]:
        raise NotImplementedError(f"Task {task} is not supported yet for {model_type}.")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    # Allocate
    pipeline_factory, model_factory = SUPPORTED_MODEL_WITH_TASKS[model_type][task]
    model = model_factory.from_pretrained(model, **kwargs)

    return pipeline_factory(model, tokenizer)
