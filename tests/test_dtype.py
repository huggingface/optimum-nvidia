import numpy as np
import pytest
import torch

import tensorrt as trt
from optimum.nvidia.lang import DataType
from tensorrt_llm._utils import np_bfloat16

@pytest.mark.parametrize(
    "literal_dtype,dtype",
    [
        ("float32", np.float32),
        ("float16", np.float16),
        ("bfloat16", np_bfloat16),
        ("int8", np.int8),
    ]
)
def test_convert_str_to_numpy(literal_dtype: str, dtype):
    assert DataType(literal_dtype).as_numpy() == dtype


@pytest.mark.parametrize(
    "literal_dtype,dtype",
    [
        ("float32", torch.float32),
        ("float16", torch.float16),
        ("float8", torch.float32),  # Change this when supported
        ("int8", torch.int8),
    ]
)
def test_convert_str_to_torch(literal_dtype: str, dtype):
    assert DataType(literal_dtype).as_torch() == dtype

@pytest.mark.parametrize(
    "literal_dtype,dtype",
    [
        ("float32", trt.float32),
        ("float16", trt.float16),
        ("float8", trt.fp8),
        ("int8", trt.int8),
    ]
)
def test_convert_str_to_tensorrt(literal_dtype: str, dtype):
    assert DataType(literal_dtype).as_trt() == dtype