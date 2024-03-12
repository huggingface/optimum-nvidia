#  coding=utf-8
#  Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import Union

import torch
from ammo.torch import export as ate
from ammo.torch import quantization as atq
from datasets import tqdm
from tensorrt_llm.quantization import QuantMode
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from transformers.quantizers import HfQuantizer
from transformers.utils.quantization_config import QuantizationConfigMixin

from optimum.nvidia.quantization.ammo import AmmoQuantizationConfig

LOGGER = getLogger(__name__)

_SUPPORTED_MODEL_ARCHITECTURES = {
    "llama": "llama",
    "mistral": "llama",
    "gemma": "gemma"
}


def get_quantization_algorithm_name(qconfig: QuantMode) -> str:
    if qconfig.has_fp8_qdq() or qconfig.has_fp8_kv_cache():
        return "fp8"
    elif qconfig.is_int4_weight_only():
        return "int4_awq"
    elif not qconfig.has_act_and_weight_quant():
        return "int8_sq"
    else:
        raise ValueError(f"Unable to determine quantization algorithm from: {qconfig}")


def infer_decoder_type(model: PreTrainedModel) -> str:
    if model.config.model_type in _SUPPORTED_MODEL_ARCHITECTURES:
        return _SUPPORTED_MODEL_ARCHITECTURES[model.config.model_type]

    else:
        raise ValueError(f"{model.config.model_type} is not supported yet")


class AmmoQuantizer(HfQuantizer):

    def __init__(
            self,
            quantization_config: QuantizationConfigMixin,
            artifact_path: Union[str, PathLike, Path],
            tensor_parallel_degree: int = 1,
            pipeline_parallel_degree: int = 1,
            export_tensorrt_llm_config: bool = True,
    ):
        if tensor_parallel_degree < 1:
            raise ValueError(f"tensor_parallel_degree should be >= 1 (got {tensor_parallel_degree})")

        if pipeline_parallel_degree < 1:
            raise ValueError(f"pipeline_parallel_degree should be >= 1 (got {pipeline_parallel_degree})")

        super().__init__(quantization_config=quantization_config)

        if not isinstance(artifact_path, Path):
            artifact_path = Path(artifact_path)

        self._artifact_path = artifact_path
        self._tp_degree = tensor_parallel_degree
        self._pp_degree = pipeline_parallel_degree
        self._export_tensorrt_llm_config = export_tensorrt_llm_config

    @property
    def is_serializable(self):
        return False

    @property
    def is_trainable(self):
        return False

    def _process_model_after_weight_loading(self, model, **kwargs):
        assert isinstance(self.quantization_config, AmmoQuantizationConfig)
        qconfig = self.quantization_config

        if qconfig.requires_calibration:
            if not qconfig.has_calibration_dataset:
                raise ValueError("Float8 quantization requires a calibration dataset")

            def _loop():
                batch_size = kwargs.pop("batch_size", 4)
                with torch.inference_mode():
                    data = DataLoader(
                        qconfig.calibration_dataset,
                        batch_size=batch_size,
                        pin_memory=True,
                        pin_memory_device="cuda:0",
                    )

                    for sample in tqdm(data):
                        inputs = {name: tensor[1].to("cuda") for name, tensor in sample.items()}
                        model(**inputs)

            model.to("cuda:0")
            atq.quantize(model, config=qconfig.as_ammo_config(), forward_loop=_loop)

    def _process_model_before_weight_loading(self, model, **kwargs):
        assert isinstance(self.quantization_config, AmmoQuantizationConfig)

        with torch.inference_mode():
            decoder_type = infer_decoder_type(model)
            ate.export_model_config(
                model=model,
                decoder_type=decoder_type,
                dtype=model.config.torch_dtype,
                export_dir=self._artifact_path,
                inference_tensor_parallel=self._tp_degree,
                inference_pipeline_parallel=self._pp_degree,
                export_tensorrt_llm_config=self._export_tensorrt_llm_config,
            )
