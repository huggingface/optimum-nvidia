import random
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
import torch
from datasets import Dataset
from modelopt.torch.export import torch_to_tensorrt_llm_checkpoint
from tensorrt_llm.quantization.quantize_by_modelopt import (
    get_model_type,
    quantize_model,
)
from torch.utils.data import DataLoader


if TYPE_CHECKING:
    from modelopt.torch.quantization import QuantizeConfig
    from tensorrt_llm import Mapping
    from torch import Tensor
    from transformers import PreTrainedModel as TransformersPreTrainedModel

    from optimum.nvidia import IntoModelOptQuantizeConfig


LOGGER = getLogger()


def quantize(
    model: "TransformersPreTrainedModel",
    qconfig: Union["QuantizeConfig", "IntoModelOptQuantizeConfig"],
    dataset: Union[Dataset, DataLoader],
    mapping: Optional["Mapping"] = None,
    seed: int = 2014,
) -> [Dict[str, Any], Dict[str, "Tensor"]]:
    if isinstance(dataset, Dataset):
        dataset = DataLoader(dataset)

    if isinstance(qconfig, IntoModelOptQuantizeConfig):
        LOGGER.info(f"Converting {qconfig} to TensorRT-LLM quantization config")
        qconfig = qconfig.into_model_opt_qconfig()

    # Seed everything
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # Retrieve additional information
    model_type = get_model_type(model)
    mapping = mapping or Mapping(
        world_size=1, rank=0, gpus_per_node=1, tp_size=1, pp_size=1
    )

    # Do the quantization
    with torch.inference_mode():
        qmodel = quantize_model(model, qconfig, dataset)

        return torch_to_tensorrt_llm_checkpoint(
            qmodel, model_type, model.dtype, mapping.tp_size, mapping.pp_size
        )
