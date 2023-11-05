from logging import getLogger
from typing import Iterable, Optional, Set, Tuple, Protocol, runtime_checkable

import numpy as np
from tensorrt_llm import Module
from tensorrt_llm.models import weight_only_groupwise_quantize
from tensorrt_llm.quantization import QuantMode


LOGGER = getLogger(__name__)


@runtime_checkable
class SupportsWeightCompression(Protocol):

    EXCLUDED_WEIGHT_PARAMETERS: Optional[Set[str]]

    @staticmethod
    @property
    def named_weight_parameters() -> Iterable[Tuple[str, np.array]]:
        ...


QUANTIZATION_PROTOCOLS = {SupportsWeightCompression}


def awq_weight_only_compression(
    model: Module,
    quantization_mode: QuantMode,
    group_size: int = 128,
    zero: bool = True,
    pre_quant_scale: bool = False,
    exclude_modules: Optional[Set[str]] = None
):
    LOGGER.debug(f"Replacing {type(model)} with AWQ specific linear modules")
    return weight_only_groupwise_quantize(
        model=model,
        quant_mode=quantization_mode,
        group_size=group_size,
        pre_quant_scale=pre_quant_scale,
        zero=zero,
        exclude_modules=exclude_modules
    )
