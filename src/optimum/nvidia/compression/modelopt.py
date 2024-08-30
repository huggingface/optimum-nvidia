from abc import ABC, abstractmethod
from typing import Protocol, Iterable, Union, TYPE_CHECKING, runtime_checkable, Optional

import modelopt.torch.quantization as mtq
import modelopt.torch.sparsity as mts
from transformers.quantizers import HfQuantizer
from transformers.utils.quantization_config import QuantizationConfigMixin

from optimum.nvidia.compression import CompressionRecipe

if TYPE_CHECKING:
    from modelopt.torch.quantization import QuantizeConfig


@runtime_checkable
class IntoModelOptQuantizeConfig(Protocol):
    def into_modelopt_qconfig(self) -> "QuantizeConfig": ...


class ModelOptConfig(QuantizationConfigMixin):
    def __init__(
            self,
            qconfig: Union["QuantizeConfig", "IntoModelOptQuantizeConfig"],
            sparsity: Optional[Union[mts.mode.SparseGPTConfig, mts.mode.SparseMagnitudeConfig]] = None):

        self._qconfig = qconfig.into_modelopt_qconfig() if isinstance(qconfig, IntoModelOptQuantizeConfig) else qconfig
        self._sparsity = sparsity

    @property
    def qconfig(self) -> "QuantizeConfig":
        return self._qconfig

    @property
    def sparsity(self) -> Optional[Union[mts.mode.SparseGPTConfig, mts.mode.SparseGPTConfig]]:
        return self._sparsity


class ModelOptRecipe(CompressionRecipe[ModelOptConfig], ABC):

    @property
    @abstractmethod
    def config(self) -> ModelOptConfig:
        raise NotImplementedError()

    @property
    @abstractmethod
    def dataset(self) -> Iterable:
        raise NotImplementedError()


class ModelOptQuantizer(HfQuantizer):

    def __init__(self, recipe: ModelOptRecipe):
        super().__init__(recipe.config)
        self._recipe = recipe

    def _process_model_before_weight_loading(self, model, **kwargs):
        pass

    def _process_model_after_weight_loading(self, model, **kwargs):
        pass

    @property
    def is_serializable(self):
        return True

    @property
    def is_trainable(self):
        return True
