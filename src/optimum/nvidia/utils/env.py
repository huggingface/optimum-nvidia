import functools
import os

from transformers.utils import strtobool

INT_TRUE_VALUE = 1


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
