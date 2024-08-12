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


from enum import Enum
from typing import List

import torch


class DataType(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT8 = "float8"
    INT64 = "int64"
    INT32 = "int32"
    INT8 = "int8"
    UINT8 = "uint8"
    BOOL = "bool"

    @staticmethod
    def from_torch(dtype: torch.dtype) -> "DataType":
        if dtype == torch.float32:
            return DataType.FLOAT32
        elif dtype == torch.float16:
            return DataType.FLOAT16
        elif dtype == torch.bfloat16:
            return DataType.BFLOAT16
        elif dtype == torch.float8_e4m3fn:
            return DataType.FLOAT8
        elif dtype == torch.int64:
            return DataType.INT64
        elif dtype == torch.int32:
            return DataType.INT32
        elif dtype == torch.int8:
            return DataType.INT8
        elif dtype == torch.uint8:
            return DataType.UINT8
        elif dtype == torch.bool:
            return DataType.BOOL
        else:
            raise ValueError(f"Unknown torch.dtype {dtype}")

    def to_trt(self) -> "DataType":
        """
        Convert textual dtype representation to their TensorRT counterpart
        :return: Converted dtype if equivalent is found
        :raise ValueError if provided dtype doesn't have counterpart
        """
        import tensorrt as trt

        if self == DataType.FLOAT32:
            return trt.DataType.FLOAT
        elif self == DataType.FLOAT16:
            return trt.DataType.HALF
        elif self == DataType.BFLOAT16:
            return trt.DataType.BF16
        elif self == DataType.FLOAT8:
            return trt.DataType.FP8
        elif self == DataType.INT8:
            return trt.DataType.INT8
        elif self == DataType.UINT8:
            return trt.DataType.UINT8
        elif self == DataType.INT32:
            return trt.DataType.INT32
        elif self == DataType.INT64:
            return trt.DataType.INT64
        elif self == DataType.BOOL:
            return trt.DataType.BOOL
        else:
            raise ValueError(f"Unknown value {self}")

    def to_torch(self):
        """
        Convert textual dtype representation to their Torch counterpart
        :return: Converted dtype if equivalent is found
        :raise ValueError if provided dtype doesn't have counterpart
        """
        import torch

        if self == DataType.FLOAT32:
            return torch.float32
        elif self == DataType.FLOAT16:
            return torch.float16
        elif self == DataType.BFLOAT16:
            return torch.bfloat16
        elif self == DataType.FLOAT8:
            return torch.float8_e4m3fn
        elif self == DataType.INT8:
            return torch.int8
        elif self == DataType.UINT8:
            return torch.uint8
        elif self == DataType.INT32:
            return torch.int32
        elif self == DataType.INT64:
            return torch.int64
        elif self == DataType.BOOL:
            return torch.bool
        else:
            raise ValueError(f"Unknown value {self}")

    @staticmethod
    def values() -> List[str]:
        return [item.value for item in DataType]
