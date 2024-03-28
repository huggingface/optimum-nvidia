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
import tempfile
from glob import glob
from pathlib import Path

import huggingface_hub
import pytest

from optimum.nvidia.models.whisper import WhisperForConditionalGeneration


@pytest.mark.parametrize(
    "model_id",
    [
        "openai/whisper-tiny",
        "openai/whisper-large-v3",
        "distil-whisper/distil-medium.en",
    ],
)
def test_whisper(model_id: str):
    # Make sure we remove the potentially already built engines.
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

        for path in [cached_path / "encoder", cached_path / "decoder"]:
            if path.exists() and path.is_dir():
                shutil.rmtree(path)
                shutil.rmtree(path)

    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    with tempfile.TemporaryDirectory() as tmp_f:
        model.save_pretrained(tmp_f)

        encoder_engines_files = glob(Path(tmp_f, "encoder/engines/*.engine").as_posix())
        decoder_engines_files = glob(Path(tmp_f, "decoder/engines/*.engine").as_posix())

        assert len(encoder_engines_files) > 0
        assert len(decoder_engines_files) > 0

        model = WhisperForConditionalGeneration.from_pretrained(tmp_f)
