from abc import ABC, abstractmethod
from typing import Protocol, Iterable, Union, TYPE_CHECKING, runtime_checkable, Optional, Sequence

import modelopt.torch.quantization as mtq
import modelopt.torch.sparsity as mts
import torch
from modelopt.torch.export import export_tensorrt_llm_checkpoint
from transformers.quantizers import HfQuantizer
from transformers.utils.quantization_config import QuantizationConfigMixin

from optimum.nvidia.compression import CompressionRecipe

if TYPE_CHECKING:
    from modelopt.torch.quantization import QuantizeConfig
    from optimum.nvidia.export import Workspace
    from transformers import PreTrainedModel as TransformersPreTrainedModel



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
    def quant_method(self):
        return self._qconfig.algorithm

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

    def _looper(self, model: "TransformersPreTrainedModel"):
        for sample in self._recipe.dataset:
            _ = model(**sample)

    def _process_model_before_weight_loading(self, model, **kwargs):
        return model

    def _process_model_after_weight_loading(self, model, **kwargs):
        if "workspace" not in kwargs:
            raise KeyError("workspace not provided but required to generate quantized model representation")

        workspace: "Workspace" = kwargs.pop("workspace")
        qmodel = mtq.quantize(model, vars(self._recipe.config.qconfig), forward_loop=self._looper)

        with torch.inference_mode():
            export_tensorrt_llm_checkpoint(
                qmodel,
                decoder_type=model.config.model_type,
                dtype=model.dtype,
                export_dir=workspace.checkpoints_path,
                inference_tensor_parallel=1,
                inference_pipeline_parallel=1,
                use_nfs_workspace=False
            )

        return qmodel

    @property
    def is_serializable(self):
        return True

    @property
    def is_trainable(self):
        return True
