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

from pathlib import Path
from typing import Dict, Optional, Type, Union

from huggingface_hub import ModelHubMixin

from optimum.nvidia.errors import UnsupportedModelException

from ..hub import extract_model_type
from .gemma import GemmaForCausalLM
from .llama import LlamaForCausalLM
from .mistral import MistralForCausalLM


_SUPPORTED_MODEL_CLASS = {
    "llama": LlamaForCausalLM,
    "mistral": MistralForCausalLM,
    "gemma": GemmaForCausalLM,
}


class AutoModelForCausalLM(ModelHubMixin):
    """ """

    @classmethod
    def _from_pretrained(
        cls: Type,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Optional[Union[str, bool]],
        **model_kwargs,
    ):
        config = model_kwargs.pop("config", None)
        if not config:
            raise ValueError("Unable to determine the model type without config")

        model_type, _ = extract_model_type(config)
        if model_type is None:
            raise ValueError(
                "Unable to determine the model type from the provided config. "
                "Please open-up an issue at https://github.com/huggingface/optimum-nvidia/issues"
            )

        if model_type not in _SUPPORTED_MODEL_CLASS:
            raise UnsupportedModelException(model_type)

        model_clazz = _SUPPORTED_MODEL_CLASS[model_type]
        model = model_clazz.from_pretrained(
            pretrained_model_name_or_path=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            **model_kwargs,
        )

        return model
