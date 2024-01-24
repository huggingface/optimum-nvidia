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
from collections import UserDict
from typing import Any, Mapping, Protocol


# TODO: Why are we not using optimum's NormalizedConfig, or transformers PretrainedConfig?
class ModelConfig(Protocol):
    @property
    def vocab_size(self) -> int:
        ...

    @property
    def max_sequence_length(self) -> int:
        ...

    @property
    def hidden_size(self) -> int:
        ...

    @property
    def intermediate_size(self) -> int:
        ...

    @property
    def num_layers(self) -> int:
        ...

    @property
    def num_heads(self) -> int:
        ...

    @property
    def use_multi_head_attention(self) -> bool:
        ...

    @property
    def activation(self) -> str:
        ...

    @property
    def num_kv_heads(self) -> int:
        ...


class TransformersConfig(UserDict, ModelConfig):
    __slots__ = ("config",)

    def __init__(self, pretrained_config: Mapping[str, Any]):
        # TODO: refactor this, this is decoder-only specific.
        if "num_heads" not in pretrained_config and "num_attention_heads" in pretrained_config:
            pretrained_config["num_heads"] = pretrained_config["num_attention_heads"]

        if "num_layers" not in pretrained_config:
            pretrained_config["num_layers"] = pretrained_config["num_hidden_layers"]

        if "max_sequence_length" not in pretrained_config:
            if "max_position_embeddings" in pretrained_config:
                pretrained_config["max_sequence_length"] = pretrained_config["max_position_embeddings"]
            elif "max_length" in pretrained_config:
                # TODO: move this whisper specific code elsewhere.
                pretrained_config["max_sequence_length"] = pretrained_config["max_length"]
            else:
                # TODO: move this elsewhere.
                raise ValueError("Unable to determine max_sequence_length from model config.")
        
        # TODO: Whisper specific. Refactor this.
        if "hidden_size" not in pretrained_config and "d_model" in pretrained_config:
            pretrained_config["hidden_size"] = pretrained_config["d_model"]

        super().__init__(pretrained_config)
        self.config = pretrained_config

    @property
    def vocab_size(self) -> int:
        return self.data["vocab_size"]

    @property
    def max_sequence_length(self) -> int:
        return self.data["max_sequence_length"]

    @property
    def hidden_size(self) -> int:
        return self.data["hidden_size"]

    @property
    def intermediate_size(self) -> int:
        return self.data["intermediate_size"]

    @property
    def num_layers(self) -> int:
        return self.data["num_hidden_layers"]

    @property
    def num_heads(self) -> int:
        return self.data["num_attention_heads"]

    @property
    def num_kv_heads(self) -> int:
        return self.get("num_key_value_heads", self.num_heads)

    @property
    def use_multi_head_attention(self) -> bool:
        return self.num_kv_heads == self.num_heads

    @property
    def activation(self) -> str:
        return self.data["hidden_act"]
