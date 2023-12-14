import functools
import os

import pytest
from distutils.util import strtobool

from optimum.nvidia.utils.nvml import get_device_count


INT_TRUE_VALUE = 1

# Environment variable controlling test set
ENVVAR_NAME_RUN_NIGHTLY = "RUN_NIGHTLY"
ENVVAR_NAME_RUN_SLOW = "RUN_SLOW"
ENVVAR_NAME_RUN_CPU_ONLY = "RUN_CPU_ONLY"


@functools.cache
def parse_flag_from_env(name: str, default: bool) -> bool:
    """
    Parse the environment variable `name` as a boolean
    :param name: Name of target environment variable
    :param default: The default value to apply if `name` is not present
    :return: Boolean value
    """

    # Retrieve the value or `default` if not present
    value = os.environ.get(name, str(default))

    try:
        return strtobool(value) == INT_TRUE_VALUE
    except ValueError:
        raise ValueError(f"Failed to convert environment variable {name}={value} to a bool")


nightly = pytest.mark.skipif(parse_flag_from_env(ENVVAR_NAME_RUN_NIGHTLY, False), reason="Nightly test")
slow = pytest.mark.skipif(parse_flag_from_env(ENVVAR_NAME_RUN_SLOW, False), reason="Slow test")

requires_gpu = pytest.mark.skipif(
    parse_flag_from_env(ENVVAR_NAME_RUN_CPU_ONLY, False) or not get_device_count(),
    reason=f"RUN_CPU_ONLY={parse_flag_from_env(ENVVAR_NAME_RUN_CPU_ONLY, False)} or "
           f"no GPU detected (num_gpus={get_device_count()})"
)

