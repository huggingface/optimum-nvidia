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


from typing import Optional

from tensorrt_llm import Mapping as Shard
from transformers import AutoModelForCausalLM

from ..configs import ModelConfig, QuantizationConfig
from .base import TensorRTEngineBuilder


class TensorRTForCausalLMEngineBuilder(TensorRTEngineBuilder):
    LOADING_CLASS = AutoModelForCausalLM

    def get_prepare_inputs_kwargs(self):
        return {
            "max_batch_size": self._optimization_profile.max_batch_size,
            "max_input_len": self._optimization_profile.max_prompt_length,
            "max_new_tokens": self._optimization_profile.max_new_tokens,
            "max_num_tokens": None,
            "max_beam_width": self._beam_width,
            "use_cache": True,
        }

    def get_builder_config_kwargs(
        self,
        config: ModelConfig,
        qconfig: QuantizationConfig,
        shard: "Shard",
        is_parallel: bool,
        opt_level: Optional[int],
    ):
        return {
            "hidden_act": config.activation,
            "num_kv_heads": config.num_kv_heads,
            "num_heads": config.num_heads,
            "max_position_embeddings": config.max_sequence_length,
            "max_input_len": self._optimization_profile.max_prompt_length,
            "max_output_len": self._optimization_profile.max_output_length,
            "max_num_tokens": None,
            "max_beam_width": self._beam_width,
            "strongly_typed": qconfig.mode.has_fp8_qdq(),
            "pipeline_parallel": shard.pp_size,
            "parallel_build": is_parallel,
            "vocab_size": config.vocab_size,
            "opt_level": opt_level,
        }
