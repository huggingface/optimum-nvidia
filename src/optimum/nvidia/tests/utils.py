import functools
import os
import unittest
from distutils.util import strtobool


INT_TRUE_VALUE = 1

# Environment variable controlling test set
ENVVAR_NAME_RUN_NIGHTLY = "RUN_NIGHTLY"
ENVVAR_NAME_RUN_SLOW = "RUN_SLOW"


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