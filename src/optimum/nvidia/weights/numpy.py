from pathlib import Path
from typing import Dict, List, Mapping, Protocol, runtime_checkable

import numpy as np
import torch
from tensorrt_llm import Module
from tensorrt_llm.quantization import QuantMode

from optimum.nvidia.lang import DataType


def as_numpy(x: torch.Tensor, dtype: DataType) -> np.array:
    x = x.to(dtype.as_torch())

    if dtype != DataType.BFLOAT16:
        x_ = x.cpu().numpy()
    else:
        x_ = x.view(torch.int16).cpu().numpy()
    return x_.view(dtype.as_numpy())


@runtime_checkable
class SupportsNpz(Protocol):
    @classmethod
    def from_numpy(cls, path: Path) -> Module:
        ...

    @staticmethod
    def get_scaling_factors(
        weights: Mapping[str, np.array], num_layers: int, mode: QuantMode
    ) -> Dict[str, List[np.array]]:
        ...
