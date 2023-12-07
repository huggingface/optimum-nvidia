import functools
from typing import NamedTuple, Optional, Tuple

from pynvml import nvmlDeviceGetCudaComputeCapability, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit


_NVML_INITIALIZED = False

MemoryInfo = NamedTuple("MemoryInfo", [
    ("total", int),
    ("free", int),
    ("used", int),
])


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


@functools.cache
@nvml_guard
def get_device_memory(device: int) -> Optional[int]:
    nvml_device_handle = nvmlDeviceGetHandleByIndex(device)
    mem_info = nvmlDeviceGetMemoryInfo(nvml_device_handle)
    return MemoryInfo(mem_info.total, mem_info.free, mem_info.used).total

@functools.cache
@nvml_guard
def get_device_count() -> int:
    import torch
    return torch.cuda.device_count()
