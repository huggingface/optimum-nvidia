from logging import getLogger
from typing import Dict, Iterable, Union
from pathlib import Path

import torch
from ammo.torch import quantization as atq, export as ate
from optimum.nvidia.configs import QuantizationConfig
from optimum.nvidia.lang import DataType, as_torch_dtype
from tensorrt_llm.builder import BuilderConfig
from tensorrt_llm.quantization import QuantMode


LOGGER = getLogger(__name__)


def get_ammo_config(mode: QuantMode, **quantizer_overrides):
    if mode.has_fp8_qdq():
        cfg = atq.FP8_DEFAULT_CFG
        if quantizer_overrides:
            for name, cfg in quantizer_overrides.items():
                cfg['quant_cfg'][name] = cfg
        return cfg
    else:
        raise NotImplementedError(
            "Only float8 quantization is supported for now. "
            "Please open up an issue on the optimum-nvidia Github with your need."
        )


def get_quantization_algorithm_name(qconfig: QuantizationConfig) -> str:
    mode = qconfig.mode

    if mode.has_fp8_qdq() or mode.has_fp8_kv_cache():
        return "fp8"
    elif mode.is_int4_weight_only():
        return "int4_awq"
    elif not mode.is_weight_only() and mode.is_int8():
        return "int8_sq"
    else:
        raise ValueError(f"Unable to determine quantization algorithm from: {qconfig}")


class AmmoQuantizer:

    def __init__(
        self,
        model: torch.nn.Module,
        qconfig: QuantizationConfig,
        dtype: DataType,
        tp_degree: int = -1,
        **quantizer_overrides
    ):
        self._model = model
        self._qconfig = qconfig
        self._dtype = dtype
        self._tp_degree = tp_degree

        # Infer the target quantization elements
        self._ammo_config = get_ammo_config(qconfig.mode, **quantizer_overrides)

    def calibrate(self, calibration_data: Iterable[Dict[str, torch.Tensor]]):
        from tqdm import tqdm
        with torch.inference_mode():
            def _loop():
                for sample in tqdm(calibration_data):
                    inputs = {name: tensor[0].view((1, -1)).to(self._model.device) for name, tensor in sample.items()}
                    self._model(**inputs)

            atq.quantize(self._model, self._ammo_config, _loop)

    def save(self, path: Union[str, Path]):
        ate.export_model_config(
            self._model,  # The quantized model.
            "llama",  # The type of the model as str, e.g gptj, llama or gptnext.
            as_torch_dtype(self._dtype),  # The exported weights data type as torch.dtype.
            get_quantization_algorithm_name(self._qconfig),  # The quantization algorithm applied, e.g. fp8 or int8_sq.
            path, # The directory where the exported files will be stored.
            self._tp_degree,  # The number of GPUs used in the inference time for tensor parallelism.
        )