from logging import getLogger
from typing import Dict, Iterable, Union
from pathlib import Path

import torch
from ammo.torch import quantization as atq, export as ate
from optimum.nvidia.configs import QuantizationConfig
from tensorrt_llm.quantization import QuantMode

LOGGER = getLogger(__name__)


def get_ammo_config(mode: QuantMode, **quantizer_overrides):
    if mode.has_fp8_qdq():
        return atq.FP8_DEFAULT_CFG
    else:
        raise NotImplementedError(
            "Only float8 quantization is supported for now. "
            "Please open up an issue on the optimum-nvidia Github with your need."
        )


class AmmoQuantizer:

    def __init__(self, model: torch.nn.Module, qconfig: QuantizationConfig, **quantizer_overrides):
        self._qconfig = qconfig
        self._model = model

        # Infer the target quantization elements
        self._ammo_config = get_ammo_config(qconfig.mode, **quantizer_overrides)

    def forward(self, **inputs):
        inputs = {name: tensor.view((1, -1)).to(self._model.device) for name, tensor in inputs.items()}
        self._model(**inputs)

    def calibrate(self, calibration_data: Iterable[Dict[str, torch.Tensor]]):
        from tqdm import tqdm
        with torch.inference_mode():
            def _loop():
                for sample in tqdm(calibration_data):
                    self.forward(**sample)

            atq.quantize(self._model, self._ammo_config, _loop)

    def save_file(self, path: Union[str, Path]):
        ate.export_model_config(
            self._model,  # The quantized model.
            "llama",  # The type of the model as str, e.g gptj, llama or gptnext.
            torch.float16,  # The exported weights data type as torch.dtype.
            "fp8",  # The quantization algorithm applied, e.g. fp8 or int8_sq.
            path, # The directory where the exported files will be stored.
            1,  # The number of GPUs used in the inference time for tensor parallelism.
        )