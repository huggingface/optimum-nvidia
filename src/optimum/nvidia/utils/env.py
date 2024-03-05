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
import os

from transformers.utils import strtobool


INT_TRUE_VALUE = 1


@functools.cache
def parse_flag_from_env(name: str, default: bool) -> bool:
    """
    Parse the environment variable `name` as a boolean
    :param name: Name of target environment variable
    :param default: The default value to apply if `name` is not present
    :return: Boolean value
    """

    # Retrieve the value or `default` if not present
    value = os.environ.get(name, str(default))

    try:
        return strtobool(value) == INT_TRUE_VALUE
    except ValueError:
        raise ValueError(
            f"Failed to convert environment variable {name}={value} to a bool"
        )
