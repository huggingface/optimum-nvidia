from logging import getLogger
from typing import Optional, Set

from tensorrt_llm import Module
from tensorrt_llm.quantization import QuantMode


LOGGER = getLogger(__name__)



def to_awq_module(
        model: Module,
        quantization_mode: QuantMode,
        group_size: int = 128,
        zero: bool = True,
        pre_quant_scale: bool = False,
        exclude_modules: Optional[Set[str]] = None
):
    """
    Apply AWQ (Activation-aware Weight Quantization) to the model's linear layers, excluding `exclude_modules`.
    :param model:
    :param quantization_mode:
    :param group_size:
    :param zero:
    :param pre_quant_scale:
    :param exclude_modules:
    :return:
    """
    LOGGER.debug(f"Replacing {type(model)} with AWQ specific linear modules")
    # return weight_only_groupwise_quantize(
    #     model=model,
    #     quant_mode=quantization_mode,
    #     group_size=group_size,
    #     pre_quant_scale=pre_quant_scale,
    #     zero=zero,
    #     exclude_modules=exclude_modules
    # )
