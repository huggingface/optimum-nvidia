import json
from pathlib import Path
from typing import Dict, Generator

import torch
from safetensors.torch import load
from transformers.modeling_utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME


def iter_safetensors(path: Path) -> Generator[Dict[str, torch.Tensor], None, None]:
    if (path / SAFE_WEIGHTS_INDEX_NAME).exists():
        with open(path / SAFE_WEIGHTS_INDEX_NAME) as index_f:
            indexes = json.load(index_f)
            for file in set(indexes["weight_map"].values()):
                with open(path / file, "rb") as shard_f:
                    yield load(shard_f.read())
    else:
        with open(path / SAFE_WEIGHTS_NAME, "rb") as safetensors_f:
            yield load(safetensors_f.read())
