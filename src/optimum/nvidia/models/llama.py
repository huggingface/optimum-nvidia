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
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from optimum.nvidia.runtime import TensorRTCompiledModel, TensorRTForCausalLM

LOGGER = getLogger(__name__)


class LlamaForCausalLM(TensorRTCompiledModel):
    __slots__ = ("_runtime", )

    def __init__(self, config: Dict[str, Any], engines_folder: Path, gpus_per_node: int, use_cuda_graph: bool = False):
        super().__init__(engines_folder)

        self._runtime = TensorRTForCausalLM(config, engines_folder, gpus_per_node, use_cuda_graph)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = -1,
        min_length: int = -1,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 0.0,
        length_penalty: float = 1.0,
        seed: int = 0,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        return self._runtime.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            seed=seed,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id
        )

