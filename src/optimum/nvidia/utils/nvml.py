import functools
from typing import Optional, Tuple

from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetCudaComputeCapability


@functools.cache
def get_device_compute_capabilities(device: int) -> Optional[Tuple[int, int]]:
    nvml_device_handle = nvmlDeviceGetHandleByIndex(device)
    return nvmlDeviceGetCudaComputeCapability(nvml_device_handle)