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
from typing import Protocol, Optional, runtime_checkable

from tensorrt_llm import Mapping
from tensorrt_llm.quantization import QuantMode


@runtime_checkable
class SupportsFromHuggingFace(Protocol):
    """
    Define the protocol implemented by TensorRT-LLM models to support loading from Hugging Face Hub
    """

    @classmethod
    def from_hugging_face(
        cls,
        hf_model_dir,
        dtype='float16',
        mapping: Optional[Mapping] = None,
        quant_mode: Optional[QuantMode] = None,
        **kwargs
    ):
        """

        :param hf_model_dir:
        :param dtype:
        :param mapping:
        :param quant_mode:
        :param kwargs:
        :return:
        """
        ...