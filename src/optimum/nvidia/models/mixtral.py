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
from os import PathLike
from typing import TYPE_CHECKING, Optional, Union

from tensorrt_llm.models.llama.model import LLaMAForCausalLM
from transformers import MixtralForCausalLM as TransformersMixtralForCausalLM

from optimum.nvidia.hub import HuggingFaceHubModel
from optimum.nvidia.models import SupportsTransformersConversion
from optimum.nvidia.runtime import CausalLM


if TYPE_CHECKING:
    from optimum.nvidia.runtime import ExecutorConfig, GenerationConfig


LOGGER = getLogger(__name__)


class MixtralForCausalLM(CausalLM, HuggingFaceHubModel, SupportsTransformersConversion):
    HF_LIBRARY_TARGET_MODEL_CLASS = TransformersMixtralForCausalLM
    TRT_LLM_TARGET_MODEL_CLASSES = LLaMAForCausalLM

    def __init__(
        self,
        engines_path: Union[str, PathLike],
        generation_config: "GenerationConfig",
        executor_config: Optional["ExecutorConfig"] = None,
    ):
        CausalLM.__init__(self, engines_path, generation_config, executor_config)
        HuggingFaceHubModel.__init__(self, engines_path)
