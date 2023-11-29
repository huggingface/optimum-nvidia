import functools

from ..version import __version__
from sys import version as pyversion

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetCudaComputeCapability, \
                    nvmlDeviceGetCount, nvmlSystemGetDriverVersion


USER_AGENT_BASE = [
    f"optimum/nvidia/{__version__}",
    f"python/{pyversion.split()[0]}"
]


@functools.cache
def get_user_agent() -> str:
    """
    Get the library user-agent when calling the hub
    :return:
    """
    ua = USER_AGENT_BASE.copy()

    try:
        nvmlInit()
        ua.append(f"nvidia/{nvmlSystemGetDriverVersion()}")


        num_gpus = nvmlDeviceGetCount()

        if num_gpus > 0:
            sm = []
            for device_idx in range(num_gpus):
                nvml_device_handle = nvmlDeviceGetHandleByIndex(device_idx)
                compute_capabilities = nvmlDeviceGetCudaComputeCapability(nvml_device_handle)
                sm.append(f"{compute_capabilities[0]}{compute_capabilities[1]}")

            ua.append(f"gpus/{num_gpus}")
            ua.append(f"sm/{'|'.join(sm)}")
    except:
        ua.append(f"nvidia/unknown")

    try:
        from torch import __version__ as pt_version
        from torch.version import cuda, cudnn
        ua.append(f"cuda/{cuda}")
        ua.append(f"cudnn/{cudnn}")
        ua.append(f"torch/{pt_version}")
    except ImportError:
        pass

    # TODO: Refactor later on
    try:
        from transformers import __version__ as tfrs_version
        ua.append(f"transformers/{tfrs_version}")
    except ImportError:
        pass

    try:
        from tensorrt_llm._utils import trt_version
        ua.append(f"tensorrt/{trt_version()}")
    except ImportError:
        pass

    return "; ".join(ua)

