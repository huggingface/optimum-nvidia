import functools
from typing import Optional, Tuple

from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetCudaComputeCapability, nvmlInit

_NVML_INITIALIZED = False


def nvml_guard(f):
    @functools.wraps(f)
    def _ensure_init(*args, **kwargs):
        global _NVML_INITIALIZED
        if not _NVML_INITIALIZED:
            nvmlInit()
            _NVML_INITIALIZED = True

        return f(*args, **kwargs)

    return _ensure_init


@functools.cache
@nvml_guard
def get_device_compute_capabilities(device: int) -> Optional[Tuple[int, int]]:
    nvml_device_handle = nvmlDeviceGetHandleByIndex(device)
    return nvmlDeviceGetCudaComputeCapability(nvml_device_handle)