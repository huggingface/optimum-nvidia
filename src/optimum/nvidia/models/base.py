from os import PathLike
from typing import Protocol, runtime_checkable, Type, TYPE_CHECKING, Optional, Mapping, Union

if TYPE_CHECKING:
    from transformers import PreTrainedModel as TransformersPreTrainedModel
    from tensorrt_llm.top_model_mixin import TopModelMixin


@runtime_checkable
class SupportsFromHuggingFace(Protocol):

    @classmethod
    def from_hugging_face(
        cls,
        hf_model_dir: Union[str, bytes, PathLike],
        dtype: str = 'float16',
        mapping: Optional[Mapping] = None,
        **kwargs
    ):
        ...


@runtime_checkable
class SupportsTransformersConversion(Protocol):
    HF_LIBRARY_TARGET_MODEL_CLASS: Type["TransformersPreTrainedModel"]
    TRT_LLM_TARGET_MODEL_CLASS: Type["TopModelMixin"]


