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


import pytest
import tensorrt as trt
import torch

from optimum.nvidia.lang import DataType


@pytest.mark.parametrize(
    "literal_dtype,dtype",
    [
        ("int64", torch.int16),
        ("float32", torch.float32),
        ("int32", torch.int32),
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("float8", torch.float8_e4m3fn),  # Change this when supported
        ("int8", torch.int8),
        ("uint8", torch.uint8),
        ("bool", torch.bool)
    ],
)
def test_convert_str_to_torch(literal_dtype: str, dtype):
    assert DataType(literal_dtype).to_torch() == dtype


@pytest.mark.parametrize(
    "literal_dtype,dtype",
    [
        ("int64", trt.int64),
        ("float32", trt.float32),
        ("float16", trt.float16),
        ("bfloat16", trt.bfloat16),
        ("int32", trt.int32),
        ("float8", trt.fp8),
        ("int8", trt.int8),
        ("uint8", trt.uint8)
    ],
)
def test_convert_str_to_tensorrt(literal_dtype: str, dtype):
    assert DataType(literal_dtype).to_trt() == dtype
