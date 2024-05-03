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
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import torch
from tensorrt_llm import Mapping
from tensorrt_llm.models import PretrainedConfig as TensorRTPretrainedConfig
from tensorrt_llm.models.modeling_utils import (
    QuantConfig as TensorRTQuantizationConfig,
)
from tensorrt_llm.plugin import PluginConfig
from tensorrt_llm.quantization import QuantMode
from transformers import AutoConfig, PretrainedConfig

from optimum.nvidia.quantization import AmmoQuantizationConfig
from optimum.nvidia.utils import get_user_agent


SUPPORTED_DTYPES = [
    "float32",
    "float16",
    "bfloat16",
    "int32",
    "int8",
    "uint8",
    "float8",
]


def dtype_to_str(dtype: Union[torch.dtype, str]) -> str:
    if isinstance(dtype, str):
        if dtype in SUPPORTED_DTYPES:
            return dtype
        else:
            raise ValueError(f"Unsupported dtype ({dtype}) value")
    elif dtype == torch.float32:
        return "float32"
    elif dtype == torch.float16:
        return "float16"
    elif dtype == torch.bfloat16:
        return "bfloat16"
    elif dtype == torch.int32:
        return "int32"
    elif dtype == torch.int8:
        return "int8"
    elif dtype == torch.uint8:
        return "uint8"
    elif dtype == torch.float8_e4m3fn:
        return "float8"
    else:
        raise ValueError(f"Unsupported torch.dtype ({dtype}) value")


def convert_quant_method_to_trt(
    method: str, weight_num_bits: int, activation_num_bits: Optional[int] = None
) -> (QuantMode, str):
    if method == "awq":
        if not activation_num_bits:
            activation_num_bits = 16

        if weight_num_bits not in {4, 8}:
            raise ValueError(
                f"Unsupported AWQ quantization schema with {weight_num_bits}-bits weights. "
                "Only 4 and 8-bits weights' quantization schemas are supported."
            )

        if activation_num_bits not in {8, 16}:
            raise ValueError(
                f"Unsupported AWQ quantization schema with {activation_num_bits}-bits activations. "
                "Only 8 and 16-bits activations' quantization schemas are supported."
            )

        mode = QuantMode.from_description(
            quantize_weights=True, per_group=True, use_int4_weights=weight_num_bits == 4
        )
        return mode, f"W{weight_num_bits}A{activation_num_bits}_AWQ"
    elif method == "gptq":
        if not activation_num_bits:
            activation_num_bits = 16

        if weight_num_bits not in {4, 8}:
            raise ValueError(
                f"Unsupported GPTQ quantization schema with {weight_num_bits}-bits weights. "
                "Only 4 and 8-bits weights' quantization schemas are supported."
            )

        if activation_num_bits != 16:
            raise ValueError(
                f"Unsupported GPTQ quantization schema with {activation_num_bits}-bits activations. "
                "Only 16-bits activations' quantization schemas are supported."
            )

        mode = QuantMode.from_description(
            quantize_weights=True,
            per_group=False,
            use_int4_weights=weight_num_bits == 4,
        )
        return mode, f"W{weight_num_bits}A16_GPTQ"
    else:
        raise ValueError(f"Unsupported quantization method: {method}")


class TensorRTConfig(ABC, TensorRTPretrainedConfig):
    @staticmethod
    def get_quantization_config(
        config: PretrainedConfig,
    ) -> (QuantMode, AmmoQuantizationConfig):
        if hasattr(config, "quantization_config"):
            qconfig = config.quantization_config
            num_bits = qconfig.num_bits
            group_size = qconfig.group_size
            mode, quant_method = convert_quant_method_to_trt(
                qconfig.quant_method, num_bits
            )
            has_zero_point = qconfig.get("zero_point", False)
            exclude_modules = qconfig.get("module_to_not_convert", [])

            return mode, TensorRTQuantizationConfig(
                quantization_algo=quant_method,
                kv_cache_quant_algo=None,
                group_size=group_size,
                has_zero_point=has_zero_point,
                exclude_modules=exclude_modules,
            )
        else:
            return QuantMode.from_description(), TensorRTQuantizationConfig(
                None, None, None, False, None
            )

    @staticmethod
    @abstractmethod
    def from_config(config: PretrainedConfig) -> "TensorRTConfig":
        raise NotImplementedError()

    @staticmethod
    def from_pretrained(
        model_id_or_path: str,
        revision: Optional[str] = None,
        token: Union[bool, str, None] = None,
    ):
        config = AutoConfig.from_pretrained(
            model_id_or_path,
            revision=revision,
            token=token,
            user_agent=get_user_agent(),
        )
        return TensorRTConfig.from_config(config)

    @staticmethod
    @abstractmethod
    def supports_strong_typing() -> bool:
        raise NotImplementedError()

    def shard(
        self,
        world_size: int,
        gpus_per_node: int,
        rank: int = 0,
        tp_degree: int = 1,
        pp_degree: int = 1,
    ):
        if tp_degree * pp_degree != world_size:
            raise ValueError(
                f"tensor parallelism ({tp_degree}) x pipeline parallelism ({pp_degree})"
                f" != world size ({world_size})"
            )

        self.mapping = Mapping(
            world_size=world_size,
            rank=rank,
            gpus_per_node=gpus_per_node,
            tp_size=tp_degree,
            pp_size=pp_degree,
        )

    def get_plugins_config(self) -> PluginConfig:
        return PluginConfig(
            rmsnorm_quantization_plugin="disable",
            layernorm_quantization_plugin="disable",
            nccl_plugin="disable",
            paged_kv_cache="enable",
            enable_xqa="enable",
            use_paged_context_fmha=None,
            use_context_fmha_for_generation=None,
            tokens_per_block=None,
            attention_qk_half_accumulation=None,
            multi_block_mode=None,
            use_custom_all_reduce=True,
            remove_input_padding=True,
            context_fmha=None,
            context_fmha_fp32_acc=None,
        )

    def save_pretrained(self, save_folder: Union[str, Path]):
        config_dict = self.to_dict()
        config_dict.pop("trt_model_class", None)
        config_dict.pop("trt_model_file", None)

        with open(Path(save_folder, "config.json"), "w") as config_f:
            json.dump(config_dict, config_f)
