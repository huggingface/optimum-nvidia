from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from modelopt.torch.quantization import QuantizeConfig


@runtime_checkable
class IntoModelOptQuantizeConfig(Protocol):
    def into_model_opt_qconfig(self) -> QuantizeConfig: ...


from .datasets import get_dataset, load_dataset, prepare_dataset
