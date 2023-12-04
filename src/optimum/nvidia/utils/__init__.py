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
import functools

from .fs import *
from .offload import maybe_offload_weights_to_cpu


def rgetattr(obj, attr):
    """
    Recursively get object attribute
    :param obj: The root object we want to retrieve nested attribute
    :param attr: The attribute path where nested attribute are comma delimited
    :return:
    """
    def _getattr(obj, attr):
        return getattr(obj, attr)
    return functools.reduce(_getattr, [obj] + attr.split('.'))