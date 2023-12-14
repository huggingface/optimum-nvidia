from abc import ABC, abstractmethod
from typing import Protocol

from optimum.nvidia.configs import NO_QUANTIZATION, QuantizationConfig
from tensorrt_llm.quantization import QuantMode


def test_no_quantization_has_quantization_step():
    qconfig = QuantizationConfig(NO_QUANTIZATION)
    assert not qconfig.has_quantization_step

def test_float8_quantization_has_quantization_step():
    qconfig = QuantizationConfig(QuantMode.from_description(
        use_fp8_qdq=True,
        use_fp8_kv_cache=True
    ))

    assert qconfig.has_quantization_step
