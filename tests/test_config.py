from typing import Optional, Tuple

import pytest
from tensorrt_llm.quantization import QuantMode

from optimum.nvidia.config import convert_quant_method_to_trt


@pytest.mark.parametrize(
    "method, weight_num_bits, activation_num_bits, expected",
    (
        "awq", 4, None, (QuantMode.from_description(quantize_weights=True, per_group=True, use_int4_weights=True), "W4A16_AWQ"),
        "awq", 8, None, (QuantMode.from_description(quantize_weights=True, per_group=True, use_int4_weights=False), "W8A16_AWQ"),
        "gptq", 4, None, (QuantMode.from_description(quantize_weights=True, per_group=False, use_int4_weights=True), "W4A16_GPTQ"),
        "gptq", 8, None, (QuantMode.from_description(quantize_weights=True, per_group=False, use_int4_weights=False), "W*A16_GPTQ")
    )
)
def test_convert_quantization_config_to_trt(
    method: str,
    weight_num_bits: int,
    activation_num_bits: Optional[int],
    expected: Tuple
):
    assert convert_quant_method_to_trt(method, weight_num_bits, activation_num_bits) == expected
