from optimum.nvidia.utils.nvml import SM_FP8_SUPPORTED

class OptimumNvidiaException(Exception):
    def __init__(self, operation: str, msg: str):
        super().__init__(f"[{operation}] {msg}.")


### Model support
class UnsupportedModelException(OptimumNvidiaException):
    def __init__(self, model_type: str):
        super(
            f"Model of type {model_type} is not supported. "
            "Please open-up an issue at https://github.com/huggingface/optimum-nvidia/issues"
        )


### Unsupported features blocks
class UnsupportedHardwareFeature(OptimumNvidiaException):
    """
    Base exception class for all features not supported by underlying hardware
    """
    def __init__(self, msg):
        super("feature", msg)

    @classmethod
    def float8(cls) -> "UnsupportedHardwareFeature":
        return Float8NotSupported()


class Float8NotSupported(UnsupportedHardwareFeature):
    """
    Thrown when attempting to target float8 inference but the underlying hardware doesn't support it
    """
    def __init__(self):
        super("float8 is not supported on your device. "
              f"Please use a device with compute capabilities {SM_FP8_SUPPORTED}")