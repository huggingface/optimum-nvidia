from typing import Iterable, Tuple, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class SupportsWeightCompression(Protocol):

    @property
    def named_weight_parameters(self) -> Iterable[Tuple[str, np.array]]:
        ...


QUANTIZATION_PROTOCOLS = {SupportsWeightCompression}