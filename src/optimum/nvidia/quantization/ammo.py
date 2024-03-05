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
from pathlib import Path
from typing import Any, Dict, Iterable, Union

import torch
from ammo.torch import export as ate
from ammo.torch import quantization as atq
from tensorrt_llm.quantization import QuantMode

from optimum.nvidia import DataType


LOGGER = getLogger(__name__)

KV_CACHE_CFG = {
    "*.query_key_value.output_quantizer": {"num_bits": 8, "axis": None, "enable": True},
    "*.Wqkv.output_quantizer": {"num_bits": 8, "axis": None, "enable": True},
    "*.W_pack.output_quantizer": {"num_bits": 8, "axis": None, "enable": True},
    "*.c_attn.output_quantizer": {"num_bits": 8, "axis": None, "enable": True},
    "*.k_proj.output_quantizer": {"num_bits": 8, "axis": None, "enable": True},
    "*.v_proj.output_quantizer": {"num_bits": 8, "axis": None, "enable": True},
}


def get_ammo_config(mode: QuantMode, quantizer_overrides):
    if mode.has_fp8_qdq():
        cfg = atq.FP8_DEFAULT_CFG

        if mode.has_fp8_kv_cache():
            kv_cache_config = KV_CACHE_CFG.copy()
            for value in kv_cache_config.values():
                value.update({"num_bits": (4, 3)})  # type: ignore
            cfg["quant_cfg"].update(kv_cache_config)

        if quantizer_overrides:
            for name, val in quantizer_overrides.items():
                cfg["quant_cfg"][name] = val

        return cfg
    else:
        raise NotImplementedError(
            "Only float8 quantization is supported for now. "
            "Please open up an issue on the optimum-nvidia Github with your need."
        )


def get_quantization_algorithm_name(qconfig: QuantMode) -> str:
    if qconfig.has_fp8_qdq() or qconfig.has_fp8_kv_cache():
        return "fp8"
    elif qconfig.is_int4_weight_only():
        return "int4_awq"
    elif not qconfig.has_act_and_weight_quant():
        return "int8_sq"
    else:
        raise ValueError(f"Unable to determine quantization algorithm from: {qconfig}")


class AmmoQuantizer:
    def __init__(
        self,
        model: torch.nn.Module,
        qconfig: QuantMode,
        dtype: DataType,
        tp_degree: int = -1,
        quantizer_overrides: Dict[str, Any] = None,
    ):
        self._model = model
        self._qconfig = qconfig
        self._dtype = dtype
        self._tp_degree = tp_degree

        # Infer the target quantization elements
        self._ammo_config = get_ammo_config(qconfig, quantizer_overrides)

    def calibrate(self, calibration_data: Iterable[Dict[str, torch.Tensor]]):
        from tqdm import tqdm

        with torch.inference_mode():

            def _loop():
                for sample in tqdm(calibration_data):
                    inputs = {
                        name: tensor[0].to("cuda") for name, tensor in sample.items()
                    }
                    self._model(**inputs)

            atq.quantize(self._model, self._ammo_config, _loop)

    def save(self, path: Union[str, Path]):
        with torch.inference_mode():
            ate.export_model_config(
                model=self._model,  # The quantized model.
                decoder_type="llama",  # The type of the model as str, e.g gptj, llama or gptnext.
                dtype=DataType(
                    self._dtype
                ).to_torch(),  # The exported weights data type as torch.dtype.
                export_dir=path,  # The directory where the exported files will be stored.
                inference_tensor_parallel=self._tp_degree,  # The number of GPUs used in the inference time for tensor parallelism.
                export_tensorrt_llm_config=True,
            )
