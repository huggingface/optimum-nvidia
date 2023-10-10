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
from mmap import mmap, ACCESS_READ, MADV_RANDOM, MADV_HUGEPAGE, MADV_WILLNEED, MADV_FREE
from os import PathLike
from typing import Protocol, runtime_checkable, Optional, Union

from huggingface_hub import cached_download
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from safetensors import safe_open
from tensorrt_llm import Mapping as ShardingConfig


LOGGER = getLogger(__name__)


def walk(path: PathLike, fs: Optional[AbstractFileSystem] = None):
    if fs is None:
        fs = LocalFileSystem()

    with fs.open(str(path), "rb") as params_f:
        LOGGER.debug(f"Opened file at {path}")

        # Memory-map the whole file
        is_linux = sys.platform == "linux"
        mm = mmap(params_f.fileno(), length=0, access=ACCESS_READ)
        if is_linux:
            LOGGER.debug("[mmap] advising MADV_RANDOM | MADV_HUGEPAGE")
            mm.madvise(MADV_RANDOM | MADV_HUGEPAGE | MADV_WILLNEED)

        # Read the content
        content = mm.read()
        with safe_open(content, framework="numpy", device="cpu") as st_content:
            for name in st_content.keys():
                yield name, st_content.get_tensor(name)

            if is_linux:
                LOGGER.debug("[mmap] advising MADV_FREE")
                mm.madvise(MADV_FREE)


@runtime_checkable
class SupportsSafetensors(Protocol):

    @classmethod
    def from_safetensors(
        cls,
        path: Union[str, PathLike],
        sharding_config: ShardingConfig,
        filesystem: Optional[AbstractFileSystem] = None
    ):
        """

        :param path:
        :param sharding_config
        :param filesystem
        :return:
        """
        ...
