import functools
import os
from distutils.util import strtobool

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
