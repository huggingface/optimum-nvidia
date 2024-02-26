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

import functools
from sys import version as pyversion

from pynvml import nvmlDeviceGetCount, nvmlInit, nvmlSystemGetDriverVersion

from optimum.nvidia.utils import parse_flag_from_env

from ..version import __version__
from .nvml import get_device_compute_capabilities


USER_AGENT_BASE = [f"optimum/nvidia/{__version__}", f"python/{pyversion.split()[0]}"]


@functools.cache
def get_user_agent() -> str:
    """
    Get the library user-agent when calling the hub
    :return:
    """
    ua = USER_AGENT_BASE.copy()

    # Nvidia driver / devices
    try:
        nvmlInit()
        ua.append(f"nvidia/{nvmlSystemGetDriverVersion()}")

        num_gpus = nvmlDeviceGetCount()
        if num_gpus > 0:
            sm = []
            for device_idx in range(num_gpus):
                compute_capabilities = get_device_compute_capabilities(device_idx)
                if compute_capabilities:
                    major, minor = compute_capabilities
                    sm.append(f"{major}{minor}")

            ua.append(f"gpus/{num_gpus}")
            ua.append(f"sm/{'|'.join(sm)}")
    except (RuntimeError, ImportError):
        ua.append("nvidia/unknown")

    # Torch / CUDA related version, (from torch)
    try:
        from torch import __version__ as pt_version
        from torch.version import cuda, cudnn

        ua.append(f"cuda/{cuda}")
        ua.append(f"cudnn/{cudnn}")
        ua.append(f"torch/{pt_version}")
    except ImportError:
        pass

    # transformers version
    try:
        from transformers import __version__ as tfrs_version

        ua.append(f"transformers/{tfrs_version}")
    except ImportError:
        pass

    # TRTLLM version
    try:
        from tensorrt_llm._utils import trt_version

        ua.append(f"tensorrt/{trt_version()}")
    except ImportError:
        pass

    # Add a flag for CI
    if parse_flag_from_env("OPTIMUM_NVIDIA_IS_CI", False):
        ua.append("is_ci/true")
    else:
        ua.append("is_ci/false")

    return "; ".join(ua)
