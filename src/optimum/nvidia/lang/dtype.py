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

import numpy as np
import torch
from tensorrt_llm import str_dtype_to_trt
from tensorrt_llm._utils import str_dtype_to_np


class DataType(Enum):
    """
    Represent the data format used to store and run actual computations
    """

    INT8 = "int8"
    FLOAT8 = "float8"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    def as_trt(self) -> object:
        return str_dtype_to_trt(self.value)

    def as_numpy(self) -> np.dtype:
        return str_dtype_to_np(self.value)

    def as_torch(self) -> "torch.dtype":
        import torch

        if self == DataType.INT8:
            return torch.int8
        elif self == DataType.BFLOAT16:
            return torch.bfloat16
        elif self == DataType.FLOAT8:
            return torch.float32  # not supported yet
        elif self == DataType.FLOAT16:
            return torch.float16
        elif self == DataType.FLOAT32:
            return torch.float32
