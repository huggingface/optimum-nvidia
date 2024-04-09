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

import shutil
from pathlib import Path

import huggingface_hub


def clean_cached_engines_for_model(model_id: str):
    cache_dir = huggingface_hub.constants.HUGGINGFACE_HUB_CACHE
    object_id = model_id.replace("/", "--")
    full_model_path = Path(cache_dir, f"models--{object_id}")
    if full_model_path.is_dir():
        # Resolve refs (for instance to convert main to the associated commit sha)
        revision_file = Path(full_model_path, "refs", "main")
        revision = ""
        if revision_file.is_file():
            with open(revision_file) as f:
                revision = f.read()
        cached_path = Path(full_model_path, "snapshots", revision)

        for path in [cached_path / "encoder", cached_path / "decoder", cached_path / "engines"]:
            if path.exists() and path.is_dir():
                shutil.rmtree(path)
