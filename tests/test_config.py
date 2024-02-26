#  coding=utf-8
#  Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


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
