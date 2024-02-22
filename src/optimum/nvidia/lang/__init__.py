from enum import Enum
from typing import List


class DataType(Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT8 = "float8"
    INT64 = "int64"
    INT32 = "int32"
    INT8 = "int8"
    UINT8 = "uint8"
    BOOL = "bool"

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