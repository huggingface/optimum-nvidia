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
from typing import Optional

from optimum.nvidia.utils.nvml import SM_FP8_SUPPORTED


class OptimumNvidiaException(Exception):
    def __init__(self, msg: str, operation: Optional[str] = None):
        if operation:
            super().__init__(f"[{operation}] {msg}.")
        else:
            super().__init__(f"{msg}")


class UnsupportedModelException(OptimumNvidiaException):
    def __init__(self, model_type: str):
        super().__init__(
            f"Model of type {model_type} is not supported. "
            "Please open-up an issue at https://github.com/huggingface/optimum-nvidia/issues"
        )


class UnsupportedHardwareFeature(OptimumNvidiaException):
    """
    Base exception class for all features not supported by underlying hardware
    """

    def __init__(self, msg, feature: str):
        super(msg)

    @classmethod
    def float8(cls) -> "UnsupportedHardwareFeature":
        return Float8NotSupported()


class Float8NotSupported(UnsupportedHardwareFeature):
    """
    Thrown when attempting to target float8 inference but the underlying hardware doesn't support it
    """

    def __init__(self):
        super().__init__(
            "float8 is not supported on your device. "
            f"Please use a device with compute capabilities {SM_FP8_SUPPORTED}",
            "float8",
        )
