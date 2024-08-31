from os import PathLike
from typing import (
    TYPE_CHECKING,
    Mapping,
    Optional,
    Protocol,
    Type,
    Union,
    runtime_checkable,
)


if TYPE_CHECKING:
    from tensorrt_llm.models import PretrainedConfig
    from tensorrt_llm.top_model_mixin import TopModelMixin
    from transformers import PreTrainedModel as TransformersPreTrainedModel


@runtime_checkable
class SupportsFromHuggingFace(Protocol):
    @classmethod
    def from_hugging_face(
        cls,
        hf_model_dir: Union[str, bytes, PathLike],
        dtype: str = "float16",
        mapping: Optional[Mapping] = None,
        **kwargs,
    ): ...


@runtime_checkable
class SupportFromTrtLlmCheckpoint(Protocol):
    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: str,
        rank: Optional[int] = None,
        config: Optional["PretrainedConfig"] = None,
    ): ...


@runtime_checkable
class SupportsTransformersConversion(Protocol):
    HF_LIBRARY_TARGET_MODEL_CLASS: Type["TransformersPreTrainedModel"]
    TRT_LLM_TARGET_MODEL_CLASSES: Union[
        Type["TopModelMixin"], Mapping[str, Type["TopModelMixin"]]
    ]
