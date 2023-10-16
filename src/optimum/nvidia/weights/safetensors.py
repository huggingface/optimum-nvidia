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
import sys
from logging import getLogger
# from mmap import mmap, ACCESS_READ, MADV_RANDOM, MADV_HUGEPAGE, MADV_WILLNEED
from os import PathLike
from typing import Protocol, runtime_checkable, Union, TypeVar

from safetensors.numpy import safe_open
from tensorrt_llm import Mapping as ShardingConfig, Module

LOGGER = getLogger(__name__)


def walk(path: PathLike):
    # with open(str(path), "rb") as params_f:
    #     LOGGER.debug(f"Opened file at {path}")

        # Memory-map the whole file
        # is_linux = sys.platform == "linux"
        # mm = mmap(params_f.fileno(), length=0, access=ACCESS_READ)
        # if is_linux:
        #     LOGGER.debug("[mmap] advising MADV_RANDOM | MADV_HUGEPAGE")
        #     mm.madvise(MADV_RANDOM | MADV_HUGEPAGE | MADV_WILLNEED)

        # Read the content
        # content = mm.read()
        # with read_safetensors(content) as st_content:

    with safe_open(path, framework="numpy") as st_content:
        LOGGER.debug(f"Opened file at {path}")
        for name in st_content.keys():
            yield name, st_content.get_tensor(name)


# Represent a generic trt module type
M_co = TypeVar("M_co", covariant=True)


@runtime_checkable
class SupportsSafetensors(Protocol[M_co]):

    @classmethod
    def from_safetensors(
        cls,
        path: Union[str, PathLike],
        sharding_config: ShardingConfig,
        model: M_co
    ):
        """

        :param path:
        :param sharding_config
        :param model
        :return:
        """
        ...
