from optimum.nvidia.errors import OptimumNvidiaException


class AlreadyQuantizedModelException(OptimumNvidiaException):
    def __init__(self):
        super().__init__(
            "Model is already quantized. Pre-quantized checkpoints are not yet supported by optimum-nvidia."
        )