from enum import Enum
from typing import List

import tensorrt as trt


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

    def to_trt(self) -> trt.DataType:
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

    @staticmethod
    def values() -> List[str]:
        return [item.value for item in DataType]

def to_trt_dtype(dtype: str) -> trt.DataType:
    """
    Convert textual dtype representation to their TensorRT counterpart
    :param dtype: Textual description of the dtype to convert
    :return: Converted dtype if equivalent is found
    :raise ValueError if provided dtype doesn't have counterpart
    """
