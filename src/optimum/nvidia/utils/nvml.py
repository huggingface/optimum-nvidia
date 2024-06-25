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
from logging import getLogger
from typing import NamedTuple, Optional, Tuple

from psutil import virtual_memory
from pynvml import (
    NVMLError,
    nvmlDeviceGetCudaComputeCapability,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName,
    nvmlInit,
)


LOGGER = getLogger()


_NVML_INITIALIZED = False

CUDA_ARCH_VOLTA = 7
CUDA_ARCH_AMPERE = 8
CUDA_ARCH_HOPPER = 9
SM_FP8_SUPPORTED = {89, 90}


MemoryInfo = NamedTuple(
    "MemoryInfo",
    [
        ("total", int),
        ("free", int),
        ("used", int),
    ],
)


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


@functools.cache
@nvml_guard
def get_device_name(device: int) -> Optional[Tuple[str, ...]]:
    try:
        nvml_device_handle = nvmlDeviceGetHandleByIndex(device)
        device_full_name = nvmlDeviceGetName(nvml_device_handle)
        return device_full_name.split()
    except NVMLError:
        return None


@functools.cache
@nvml_guard
def has_float8_support() -> bool:
    compute_capabilities = get_device_compute_capabilities(0)
    if compute_capabilities:
        compute_capabilities_ = compute_capabilities[0] * 10 + compute_capabilities[1]

        return compute_capabilities_ in SM_FP8_SUPPORTED
    else:
        LOGGER.warning(
            "Failed to retrieve the proper compute capabilities on the device"
        )
        return False


@functools.cache
@nvml_guard
def is_post_volta() -> bool:
    major, _ = get_device_compute_capabilities(0)
    return major >= CUDA_ARCH_VOLTA


@functools.cache
@nvml_guard
def is_post_ampere() -> bool:
    major, _ = get_device_compute_capabilities(0)
    return major >= CUDA_ARCH_AMPERE


@functools.cache
@nvml_guard
def is_post_hopper() -> bool:
    major, _ = get_device_compute_capabilities(0)
    return major >= CUDA_ARCH_HOPPER


def get_max_memory():
    fraction_device_map = {
        device_id: get_device_memory(device_id) * 0.7
        for device_id in range(get_device_count())
    }

    cpu_device_map = {"cpu": virtual_memory().available * 0.8}

    return fraction_device_map | cpu_device_map
