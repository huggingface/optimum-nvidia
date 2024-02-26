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
from optimum.nvidia.utils.nvml import get_device_count


# Environment variable controlling test set
ENVVAR_NAME_RUN_NIGHTLY = "RUN_NIGHTLY"
ENVVAR_NAME_RUN_SLOW = "RUN_SLOW"
ENVVAR_NAME_RUN_CPU_ONLY = "RUN_CPU_ONLY"


nightly = pytest.mark.skipif(parse_flag_from_env(ENVVAR_NAME_RUN_NIGHTLY, False), reason="Nightly test")
slow = pytest.mark.skipif(parse_flag_from_env(ENVVAR_NAME_RUN_SLOW, False), reason="Slow test")

requires_gpu = pytest.mark.skipif(
    parse_flag_from_env(ENVVAR_NAME_RUN_CPU_ONLY, False) or not get_device_count(),
    reason=f"RUN_CPU_ONLY={parse_flag_from_env(ENVVAR_NAME_RUN_CPU_ONLY, False)} or "
    f"no GPU detected (num_gpus={get_device_count()})",
)
