import functools
import os
import unittest
from distutils.util import strtobool

from optimum.nvidia.utils.nvml import get_device_count


INT_TRUE_VALUE = 1

# Environment variable controlling test set
ENVVAR_NAME_RUN_NIGHTLY = "RUN_NIGHTLY"
ENVVAR_NAME_RUN_SLOW = "RUN_SLOW"
ENVVAR_NAME_RUN_CPUONLY = "RUN_CPUONLY"


@functools.cache
def parse_flag_from_env(name: str, default: bool) -> bool:
    """
    Parse the environment variable `name` as a boolean
    :param name: Name of target environment variable
    :param default: The default value to apply if `name` is not present
    :return: Boolean value
    """

    # Retrieve the value or `default` if not present
    value = os.environ.get(name, default)

    try:
        return strtobool(value) == INT_TRUE_VALUE
    except ValueError:
        raise ValueError(f"Failed to convert environment variable {name}={value} to a bool")


def nightly(f):
    """
    Mark this callable as a nightly callable for the CI
    :param f:
    :return:
    """
    return unittest.skipUnless(parse_flag_from_env(ENVVAR_NAME_RUN_NIGHTLY), "test is nightly")(f)

def slow(f):
    """
    Mark this callable as a slow callable for the CI
    :param f:
    :return:
    """
    return unittest.skipUnless(parse_flag_from_env(ENVVAR_NAME_RUN_SLOW), "test is slow")(f)

def requires_gpu(count: int = 1):
    """
    Ensure the unittest callable will execute on as many GPUs as `count`
    :param count: Number of required GPUs
    :return:
    """
    assert count >= 1, f"count should be >= 1, got {count}"
    cpu_only = parse_flag_from_env(ENVVAR_NAME_RUN_CPUONLY)

    def _checked_gpu_decorator(f):
        num_gpus = get_device_count()
        return unittest.skipUnless(
            not cpu_only and num_gpus >= count,
            f"Not enough GPU (requested: {count}, available: {num_gpus})"
        )(f)
    return _checked_gpu_decorator
