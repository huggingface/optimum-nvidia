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
from itertools import chain
from logging import getLogger
from mmap import ACCESS_READ, mmap
from os import PathLike
from sys import platform
from typing import Iterable, List, Mapping, Protocol, TypeVar, Union, runtime_checkable

import numpy as np
from safetensors.numpy import load
from tensorrt_llm import BuilderConfig, Module
from tensorrt_llm import Mapping as ShardingConfig

from optimum.nvidia.configs import ModelConfig, QuantizationConfig


LOGGER = getLogger(__name__)


class SafetensorsAccessor(Mapping[str, np.array]):
    __slots__ = ("_buffers", "_indexes")

    @classmethod
    def from_files(cls, files: Iterable[PathLike]):
        buffers = []
        for path in files:
            with open(path, mode="rb") as fd:
                # Memory-map the whole file
                is_linux = platform == "linux"
                with mmap(fd.fileno(), length=0, access=ACCESS_READ) as mm:
                    if is_linux:
                        from mmap import MADV_HUGEPAGE, MADV_SEQUENTIAL, MADV_WILLNEED

                        LOGGER.debug("[mmap] advising MADV_RANDOM | MADV_HUGEPAGE | MADV_WILLNEED")
                        mm.madvise(MADV_SEQUENTIAL | MADV_HUGEPAGE | MADV_WILLNEED)

                    # Append the file descriptor and memory mapped handle
                    buffers.append(load(mm.read()))
        return cls(buffers)

    def __init__(self, buffers: Iterable[Mapping[str, np.array]]):
        self._buffers = buffers
        self._indexes = {name: buffer for buffer in buffers for name in buffer.keys()}

    def __getitem__(self, __key):
        buffer = self._indexes[__key]
        return buffer[__key]

    def __len__(self):
        return len(self._indexes)

    def __iter__(self):
        return chain(self._buffers)


# Represent a generic trt module type
M_co = TypeVar("M_co", covariant=True)


@runtime_checkable
class SupportsSafetensors(Protocol[M_co]):
    @classmethod
    def from_safetensors(
        cls,
        paths: List[Union[str, PathLike]],
        model: M_co,
        config: ModelConfig,
        builder_config: BuilderConfig,
        qconfig: QuantizationConfig,
        sharding_config: ShardingConfig,
    ) -> Module:
        """

        :param paths:
        :param model:
        :param config:
        :param builder_config:
        :param qconfig:
        :param sharding_config
        :return:
        """
        ...
