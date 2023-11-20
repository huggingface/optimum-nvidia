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
import os
from functools import singledispatch
from os import PathLike
from typing import Union

from huggingface_hub import HfFileSystem, hf_hub_download


@singledispatch
def ensure_file_exists_locally(_, root: Union[str, PathLike], file: Union[str, PathLike]) -> Union[str, PathLike]:
    return os.path.join(root, file)


@ensure_file_exists_locally.register
def _(_: HfFileSystem, repo_id: Union[str, PathLike], file: Union[str, PathLike]) -> Union[str, PathLike]:
    return hf_hub_download(repo_id, filename=file)