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
import json
import os
from logging import getLogger
from os import PathLike
from typing import Iterable, Union

from fsspec import AbstractFileSystem


LOGGER = getLogger(__name__)
SAFETENSORS_WEIGHT_FILENAME = "model.safetensors"
SAFETENSORS_INDEX_FILENAME = f"{SAFETENSORS_WEIGHT_FILENAME}.index.json"


def get_safetensors_files(fs: AbstractFileSystem, model_id_or_path: Union[str, PathLike]) -> Iterable[str]:
    # Check if the model is sharded by looking at the model.safetensors.index.json
    if fs.exists(os.path.join(model_id_or_path, SAFETENSORS_INDEX_FILENAME)):
        LOGGER.debug("Found sharded safetensors model checkpoint")

        # Retrieve the index
        with fs.open(os.path.join(model_id_or_path, SAFETENSORS_INDEX_FILENAME), "r") as index_f:
            index_content = json.load(index_f)
            weight_map = index_content["weight_map"]
            shards = set(weight_map.values())

            LOGGER.debug(f"Detected {len(shards)} shards for {model_id_or_path}")
            return shards

    elif fs.exists(os.path.join(model_id_or_path, SAFETENSORS_WEIGHT_FILENAME)):
        LOGGER.debug("Found safetensors model checkpoint")
        return [SAFETENSORS_WEIGHT_FILENAME]

    else:
        LOGGER.error(f"Cannot find safetensors checkpoint for {model_id_or_path}")
        # TODO: better error for gated model!
        raise FileNotFoundError(f"Cannot find safetensors checkpoint for {model_id_or_path}")
