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

from tensorrt_llm.models.gemma.model import GemmaForCausalLM as TrtGemmaForCausalLM
from transformers import GemmaForCausalLM as TransformersGemmaForCausalLM

from optimum.nvidia.hub import HuggingFaceHubModel
from optimum.nvidia.models import SupportsTransformersConversion
from optimum.nvidia.runtime import CausalLM


LOGGER = getLogger(__name__)


class GemmaForCausalLM(CausalLM, HuggingFaceHubModel, SupportsTransformersConversion):
    HF_LIBRARY_TARGET_MODEL_CLASS = TransformersGemmaForCausalLM
    TRT_LLM_TARGET_MODEL_CLASSES = TrtGemmaForCausalLM

    TRT_LLM_MANDATORY_CONVERSION_PARAMS = {"share_embedding_table": True}
