from typing import NamedTuple

from tensorrt_llm.quantization import QuantMode


QuantizationConfig = NamedTuple("QuantizationConfig", [
    ("mode", QuantMode),
    ("group_size", int),
])
