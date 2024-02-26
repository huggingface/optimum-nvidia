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
from typing import Dict

import numpy as np
from tensorrt_llm.models import PretrainedConfig, PretrainedModel
from tensorrt_llm.models.llama.model import LLaMAForCausalLM
from tensorrt_llm.models.llama.weight import load_from_hf_llama
from tensorrt_llm.plugin import PluginConfig
from transformers import LlamaForCausalLM as TransformersLlamaForCausalLM
from transformers import PretrainedConfig as TransformersPretrainedConfig
from transformers import PreTrainedModel as TransformersPretrainedModel

from optimum.nvidia import TensorRTConfig
from optimum.nvidia.config import dtype_to_str
from optimum.nvidia.hub import HuggingFaceHubModel
from optimum.nvidia.runtime import CausalLM


LOGGER = getLogger(__name__)


class LlamaConfig(TensorRTConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`TensorRTConfig`] and can be used to control the model outputs. Read the
    documentation from [`TensorRTConfig`] for more information.
    """

    @staticmethod
    def from_config(config: TransformersPretrainedConfig) -> "TensorRTConfig":
        # Retrieve the quantization from the transformers config (if provided)
        qmode, qconfig = TensorRTConfig.get_quantization_config(config)

        trt_config = LlamaConfig(
            architecture=config.architectures[0],
            dtype=dtype_to_str(config.torch_dtype),
            logits_dtype="float32",
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            hidden_act=config.hidden_act,
            intermediate_size=config.intermediate_size,
            norm_epsilon=config.rms_norm_eps,
            position_embedding_type="rope_gpt_neox",
            world_size=1,
            tp_size=1,
            pp_size=1,
            quant_mode=qmode,
            quant_kwargs=qconfig.to_dict(),
            use_prompt_tuning=False,
            use_parallel_embedding=False,
            embedding_sharding_dim=0,
            share_embedding_table=False,
            max_lora_rank=64,
            head_size=config.hidden_size / config.num_attention_heads,
        )

        trt_config.mapping.gpus_per_node = min(trt_config.mapping.world_size, 8)

        return trt_config

    def get_plugins_config(self) -> PluginConfig:
        config = super().get_plugins_config()
        config.moe_plugin = "disable"  # TODO : Mixtral?
        config.bert_attention_plugin = "disable"
        config.gpt_attention_plugin = self.dtype
        config.gemm_plugin = self.dtype

        return config

    @staticmethod
    def supports_strong_typing() -> bool:
        return False


class LlamaForCausalLM(CausalLM, HuggingFaceHubModel):
    MODEL_CONFIG = LlamaConfig
    HF_LIBRARY_TARGET_MODEL_CLASS = TransformersLlamaForCausalLM
    TRT_LLM_TARGET_MODEL_CLASS = LLaMAForCausalLM

    @staticmethod
    def convert_weights(
        target: PretrainedModel, source: TransformersPretrainedModel, config: PretrainedConfig
    ) -> Dict[str, np.ndarray]:
        if config.quant_mode.has_any_quant():
            raise NotImplementedError("Quantization is not supported yet.")

        return load_from_hf_llama(target, source, config.mapping, config.dtype)
