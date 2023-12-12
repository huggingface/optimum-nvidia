from typing import Protocol, Type

from huggingface_hub import ModelHubMixin


class Pipeline(Protocol):
    TARGET_FACTORY: Type[ModelHubMixin]
