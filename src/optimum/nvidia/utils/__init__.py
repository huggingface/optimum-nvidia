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
import functools

from .constants import (
    DEFAULT_ENGINE_FOLDER,
    DEFAULT_HF_HUB_TRT_REVISION,
    OPTIMUM_NVIDIA_CONFIG_FILE,
    OPTIMUM_NVIDIA_CONFIG_FILENAME,
    TENSORRT_ENGINE_EXT,
    TENSORRT_TIMINGS_FILE,
)
from .env import parse_flag_from_env
from .hub import get_user_agent, model_type_from_known_config
from .nvml import has_float8_support
from .offload import maybe_offload_weights_to_cpu
from .onnx import to_onnx


def rgetattr(obj, attr):
    """
    Recursively get object attribute
    :param obj: The root object we want to retrieve nested attribute
    :param attr: The attribute path where nested attribute are comma delimited
    :return:
    """

    def _getattr(obj, attr):
        return getattr(obj, attr)

    return functools.reduce(_getattr, [obj] + attr.split("."))
