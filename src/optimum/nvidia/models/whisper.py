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
from typing import (
    TYPE_CHECKING,
)

from tensorrt_llm.models import DecoderModel as TrtDecoderModel
from tensorrt_llm.models import WhisperEncoder as TrtWhisperEncoder
from transformers.models.whisper.modeling_whisper import (
    WhisperForConditionalGeneration as TransformersWhisperForConditionalGeneration,
)

from optimum.nvidia.models import SupportsTransformersConversion


if TYPE_CHECKING:
    pass


LOGGER = getLogger(__name__)


class WhisperForConditionalGeneration(SupportsTransformersConversion):
    HF_LIBRARY_TARGET_MODEL_CLASS = TransformersWhisperForConditionalGeneration
    TRT_LLM_TARGET_MODEL_CLASSES = {
        "encoder": TrtWhisperEncoder,
        "decoder": TrtDecoderModel,
    }
