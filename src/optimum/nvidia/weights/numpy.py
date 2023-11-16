import numpy as np
from pathlib import Path
from typing import Dict, List, Mapping, Protocol, runtime_checkable

from tensorrt_llm import Module
from tensorrt_llm.quantization import QuantMode


@runtime_checkable
class SupportsNpz(Protocol):

    @classmethod
    def from_numpy(cls, path: Path) -> Module:
        ...

    @staticmethod
    def get_scaling_factors(weights: Mapping[str, np.array], num_layers: int, mode: QuantMode) -> Dict[str, List[np.array]]:
        ...