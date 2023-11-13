from .base import SupportsWeightQuantization, QUANTIZATION_PROTOCOLS
from .awq import to_awq_module

from tensorrt_llm.quantization import QuantMode

NO_QUANTIZATION = QuantMode(0)