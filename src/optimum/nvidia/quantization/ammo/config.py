import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from ammo.torch import quantization as atq
from datasets import Dataset
from transformers import PreTrainedTokenizer
from transformers.utils.quantization_config import QuantizationConfigMixin

from optimum.nvidia.lang import DataType
from optimum.nvidia.quantization.datasets import get_dataset


dtype = Union[str, torch.dtype]
TORCH_FLOAT8 = {torch.float8_e4m3fn, torch.float8_e5m2}


KV_CACHE_CFG = {
    "*.query_key_value.output_quantizer": {"num_bits": 8, "axis": None, "enable": True},
    "*.Wqkv.output_quantizer": {"num_bits": 8, "axis": None, "enable": True},
    "*.W_pack.output_quantizer": {"num_bits": 8, "axis": None, "enable": True},
    "*.c_attn.output_quantizer": {"num_bits": 8, "axis": None, "enable": True},
    "*.k_proj.output_quantizer": {"num_bits": 8, "axis": None, "enable": True},
    "*.v_proj.output_quantizer": {"num_bits": 8, "axis": None, "enable": True},
}


class QuantizationMethod(str, Enum):
    FLOAT8 = "fp8"


class AmmoQuantizationConfig(ABC, QuantizationConfigMixin):
    def __init__(
        self,
        quant_method: QuantizationMethod,
        with_quantized_kv_cache: bool = False,
        with_quantized_lm_head: bool = False,
        calibration_data: Optional[Dataset] = None,
    ):
        super().__init__(quant_method)

        self._with_quantized_kv_cache: bool = with_quantized_kv_cache
        self._with_quantized_lm_head: bool = with_quantized_lm_head
        self._calibration_dataset = calibration_data

    @property
    @abstractmethod
    def weight_dtype(self) -> torch.dtype:
        raise NotImplementedError("AmmoQuantizationConfig::weight_dtype is abstract.")

    @property
    @abstractmethod
    def has_quantized_kv_cache(self) -> bool:
        raise NotImplementedError(
            "AmmoQuantizationConfig::has_quantized_kv_cache is abstract."
        )

    @property
    def has_calibration_dataset(self) -> bool:
        return self._calibration_dataset is not None

    @property
    @abstractmethod
    def requires_calibration(self) -> bool:
        raise NotImplementedError(
            "AmmoQuantizationConfig::requires_calibration is abstract."
        )

    @property
    def calibration_dataset(self) -> Optional[Dataset]:
        return self._calibration_dataset

    @abstractmethod
    def as_ammo_config(self) -> Dict[str, Any]:
        raise NotImplementedError("AmmoQuantizationConfig::as_ammo_config is abstract.")


class Float8QuantizationConfig(AmmoQuantizationConfig):
    def __init__(
        self,
        with_quantized_kv_cache: bool = False,
        with_quantized_lm_head: bool = False,
        calibration_data: Optional[Dataset] = None,
    ):
        super().__init__(
            QuantizationMethod.FLOAT8,
            with_quantized_kv_cache,
            with_quantized_lm_head,
            calibration_data,
        )

    @property
    def weight_dtype(self) -> torch.dtype:
        return torch.float8_e4m3fn

    @property
    def has_quantized_kv_cache(self) -> bool:
        return self._with_quantized_kv_cache

    @property
    def has_quantized_lm_head(self) -> bool:
        return self._with_quantized_lm_head

    @property
    def requires_calibration(self) -> bool:
        return True

    def as_ammo_config(self) -> Dict[str, Any]:
        cfg = atq.FP8_DEFAULT_CFG.copy()
        quant_config = cfg["quant_cfg"]

        if self.has_quantized_kv_cache:
            quant_kv_cache_config = KV_CACHE_CFG.copy()
            for value in quant_kv_cache_config.values():
                value.update(num_bits=(4, 3))

            quant_config.update(quant_kv_cache_config)

        quant_config["*lm_head*"] = {"enable": self.has_quantized_lm_head}

        return cfg


class AutoQuantizationConfig:
    @classmethod
    def from_dict(cls, kwargs):
        return cls.from_description(**kwargs)

    @classmethod
    def from_description(
        cls,
        weight: dtype,
        activation: dtype,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        dataset: Optional[Union[str, Dataset]] = None,
        split: Optional[str] = "train",
        num_samples: int = 512,
        max_sequence_length: int = 1024,
        seed: int = 2016,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        if isinstance(weight, str):
            weight = DataType(weight).to_torch()

        if isinstance(activation, str):
            activation = DataType(activation).to_torch()

        if isinstance(dataset, str):
            dataset = get_dataset(
                dataset,
                tokenizer,
                num_samples,
                seqlen=max_sequence_length,
                split=split,
                seed=seed,
            )
        else:
            raise ValueError("Providing custom dataset is not yet supported")

        # float8 case
        if weight in TORCH_FLOAT8:
            return Float8QuantizationConfig(
                with_quantized_kv_cache=activation in TORCH_FLOAT8,
                with_quantized_lm_head=False,
                calibration_data=dataset,
            )
        else:
            raise NotImplementedError(
                f"Quantization(weight= {weight}, activation={activation}) is not supported yet."
            )
