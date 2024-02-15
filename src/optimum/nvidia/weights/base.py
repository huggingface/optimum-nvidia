#  coding=utf-8
#  Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
from abc import ABC, abstractmethod
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import List, Mapping, Tuple, Union

import numpy as np
import torch
from tensorrt_llm import BuilderConfig, Module
from tensorrt_llm import Mapping as ShardingConfig
from tensorrt_llm.quantization import QuantMode

from optimum.nvidia.configs import ModelConfig, QuantizationConfig
from optimum.nvidia.lang import DataType

from .numpy import as_numpy
from .safetensors import SafetensorsAccessor


LOGGER = getLogger(__name__)


def repeat_heads(tensor: np.array, factor: int, axis: int) -> np.array:
    """
    Repeat `factor` number of times the elements over the specified axis and return the new array
    :param tensor: The original tensor that needs to see its axis repeated
    :param factor: The number of time we should repeat elements over `axis`
    :param axis: Over which axis elements will be repeated
    :return: Tensor with `factor`-repeated axis
    """
    tensor_ = np.expand_dims(tensor, axis=axis).repeat(factor, axis=axis)
    return tensor_.reshape(-1, tensor.shape[-1])


def pack_qkv(
    num_layers: int,
    layer_prefix: str,
    attn_layer_name: str,
    weights: Mapping[str, np.array],
    precision: DataType,
    use_multi_head_attention: bool,
    num_kv_heads: int,
    shard_info: ShardingConfig,
) -> List[Tuple]:
    qkv_packed_layers = []
    for layer_idx in range(num_layers):
        prefix = f"{layer_prefix}.{layer_idx}.{attn_layer_name}."

        # Merge QKV
        q_weight = as_numpy(weights[prefix + "q_proj.weight"], precision)
        k_weight = as_numpy(weights[prefix + "k_proj.weight"], precision)
        v_weight = as_numpy(weights[prefix + "v_proj.weight"], precision)

        if not use_multi_head_attention:
            if num_kv_heads < shard_info.tp_size:
                LOGGER.debug(f"Duplicating KV heads ({num_kv_heads}) up to TP-degree ({shard_info.tp_size})")

                factor = shard_info.tp_size // num_kv_heads
                k_weight = repeat_heads(k_weight, factor, axis=1)
                v_weight = repeat_heads(v_weight, factor, axis=1)

        # At least one of the query, key, value projection has bias.
        if prefix + "q_proj.bias" in weights or prefix + "k_proj.bias" in weights or prefix + "v_proj.bias" in weights:
            # For example whisper encoder k_proj does not have bias, so we will just fill with zeros the fused bias if needed.
            numpy_precision = precision.as_numpy()
            if prefix + "q_proj.bias" in weights:
                q_bias = as_numpy(weights[prefix + "q_proj.bias"], precision)
            else:
                q_bias = np.zeros(q_weight.shape[0], dtype=numpy_precision)

            if prefix + "k_proj.bias" in weights:
                k_bias = as_numpy(weights[prefix + "k_proj.bias"], precision)
            else:
                k_bias = np.zeros(k_weight.shape[0], dtype=numpy_precision)

            if prefix + "v_proj.bias" in weights:
                v_bias = as_numpy(weights[prefix + "v_proj.bias"], precision)
            else:
                v_bias = np.zeros(v_weight.shape[0], dtype=numpy_precision)

            qkv_bias = (q_bias, k_bias, v_bias)
        else:
            qkv_bias = None

        qkv_weight = (q_weight, k_weight, v_weight)

        # Insert the packed weights inside the weights
        qkv_packed_layers.append((qkv_weight, qkv_bias))

    return qkv_packed_layers


class WeightAdapter(ABC):
    """ """

    __slots__ = ("_sharding_config",)
    TENSORRT_LLM_MODEL_CLASS = None

    def __init__(self, sharding_config: ShardingConfig):
        self._sharding_config = sharding_config

    @abstractmethod
    def convert(
        self,
        model: Module,
        config: ModelConfig,
        builder: BuilderConfig,
        qconfig: QuantizationConfig,
        rank: int,
        weights: Mapping[str, torch.Tensor],
    ) -> Module:
        """

        :param model:
        :param config
        :param builder
        :param qconfig
        :param rank:
        :param weights
        :return:
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def allocate_model(conf: ModelConfig, sharding: ShardingConfig, dtype: DataType, quant_mode: QuantMode) -> Module:
        """

        :param conf:
        :param sharding
        :param dtype
        :param quant_mode
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def from_safetensors(
        cls,
        paths: List[Union[str, PathLike]],
        model: Module,
        config: ModelConfig,
        builder_config: BuilderConfig,
        qconfig: QuantizationConfig,
        sharding_config: ShardingConfig,
    ) -> Module:
        if not isinstance(model, cls.TENSORRT_LLM_MODEL_CLASS):
            raise ValueError(
                f"The argument `model` to the method {cls.__name__}.from_safetensors has to be a derived type from TensorRT-LLM's {cls.TENSORRT_LLM_MODEL_CLASS.__name__}, got {type(model)}."
            )

        accessor = SafetensorsAccessor.from_files(paths)
        adapter = cls(sharding_config)
        return adapter.convert(model, config, builder_config, qconfig, sharding_config.rank, accessor)

    @classmethod
    def from_numpy(cls, path: Path) -> Module:
        # TODO: Currently only used to load quantized models, might need to change later on
        return np.load(path, "r", allow_pickle=False)
