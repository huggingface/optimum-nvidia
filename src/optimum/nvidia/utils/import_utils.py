import importlib.util


def is_tensorrt_llm_available() -> bool:
    return importlib.util.find_spec("tensorrt_llm") is not None
