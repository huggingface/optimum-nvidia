#  coding=utf-8
#  Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
from enum import Enum
from functools import singledispatch


class DataType(Enum):
    """
    Represent the data format used to store and run actual computations
    """
    INT8 = "int8"
    FLOAT8 = "float8"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"




import torch


@singledispatch
def as_torch_dtype(dtype) -> torch.dtype:
    pass


@as_torch_dtype.register
def _(dtype: str) -> torch.dtype:
    dtype_ = dtype.lower()
    if dtype_ == "float32":
        return torch.float32
    elif dtype_ == "float16":
        return torch.float16
    elif dtype_ == "bfloat16":
        return torch.bfloat16
    elif dtype_ == "int8":
        return torch.int8
    elif dtype_ == "int4":
        return torch.quint4x2
    elif dtype_ == "fp8":
        return torch.float32  # not supported yet
    else:
        raise ValueError(f"Unsupported dtype: {dtype_} to PyTorch")


@as_torch_dtype.register
def _(dtype: DataType) -> torch.dtype:
    if dtype == DataType.INT8:
        return torch.int8
    elif dtype == DataType.BFLOAT16:
        return torch.bfloat16
    elif dtype == DataType.FLOAT8:
        return torch.float32  # not supported yet
    elif dtype == DataType.FLOAT16:
        return torch.float16
    elif dtype == DataType.FLOAT32:
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype} to PyTorch")
