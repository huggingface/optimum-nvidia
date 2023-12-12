from enum import Enum


class Operation(Enum):
    QUANTIZATION = "quantization"


class UnsupportedOperation(Exception):
    def __init__(self, operation: Operation, msg: str):
        super().__init__(f"{operation.value} is not supported ({msg}).")

    @classmethod
    def quantization(cls, msg: str) -> "UnsupportedOperation":
        return UnsupportedOperation(Operation.QUANTIZATION, msg)
