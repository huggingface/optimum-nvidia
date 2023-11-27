import cachetools

from ..version import __version__
from sys import version as pyversion


USER_AGENT_BASE = [f"optimum/nvidia/{__version__}", f"python/{pyversion.split()[0]}"]


@cachetools.cached
def get_user_agent() -> str:
    """
    Get the library user-agent when calling the hub
    :return:
    """
    ua = USER_AGENT_BASE.copy()

    # TODO: Refactor later on
    try:
        from transformers import __version__ as tfrs_version
        ua = ua.append(f"transformers/{tfrs_version}")
    except ImportError:
        pass

    try:
        from torch import __version__ as pt_version
        ua = ua.append(f"torch/{pt_version}")
    except ImportError:
        pass

    try:
        from tensorrt_llm._utils import trt_version
        ua = ua.append(f"tensorrt/{trt_version()}")
    except ImportError:
        pass

    return "; ".join(ua)

