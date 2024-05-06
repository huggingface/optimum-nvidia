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

import pytest

from optimum.nvidia.utils import parse_flag_from_env
from optimum.nvidia.utils.constants import SM_ADA_LOVELACE
from optimum.nvidia.utils.nvml import get_device_compute_capabilities, get_device_count


# Environment variable controlling test set
ENVVAR_NAME_RUN_NIGHTLY = "RUN_NIGHTLY"
ENVVAR_NAME_RUN_SLOW = "RUN_SLOW"
ENVVAR_NAME_RUN_CPU_ONLY = "RUN_CPU_ONLY"


nightly = pytest.mark.skipif(
    parse_flag_from_env(ENVVAR_NAME_RUN_NIGHTLY, False), reason="Nightly test"
)
slow = pytest.mark.skipif(
    parse_flag_from_env(ENVVAR_NAME_RUN_SLOW, False), reason="Slow test"
)

requires_gpu = pytest.mark.skipif(
    parse_flag_from_env(ENVVAR_NAME_RUN_CPU_ONLY, False) or not get_device_count(),
    reason=f"RUN_CPU_ONLY={parse_flag_from_env(ENVVAR_NAME_RUN_CPU_ONLY, False)} or "
    f"no GPU detected (num_gpus={get_device_count()})",
)

requires_multi_gpu = pytest.mark.skipif(
    get_device_count() < 2, reason="At least two GPUs are required"
)


def requires_gpu_compute_capabilities_ge(min_capabilities: int, device: int = 0):
    (major, minor) = get_device_compute_capabilities(device)
    compute_capabilities = major * 10 + minor
    return pytest.mark.skipif(
        compute_capabilities < min_capabilities,
        reason=f"Require compute capabilities >= sm_{min_capabilities} but current GPU is sm_{compute_capabilities}",
    )


requires_float8 = requires_gpu_compute_capabilities_ge(SM_ADA_LOVELACE)
