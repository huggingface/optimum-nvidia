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
from pathlib import Path
from typing import Dict, Optional, Type, Union

from huggingface_hub import ModelHubMixin
from tensorrt_llm import Module

from optimum.nvidia.weights import WeightAdapter


class ConvertibleModel:
    """

    """
    ADAPTER: Type[WeightAdapter]
    TARGET: Type[Module]



class AutoModelForCausalLM(ModelHubMixin):

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

        model_type = config["model_type"]
        if model_type == "llama":
            # TODO: Fix circular import
            from .llama import LLamaForCausalLM
            model_class = LLamaForCausalLM
        else:
            raise NotImplementedError(f"Model architecture {model_type} is not supported yet.")

        return model_class.from_pretrained(
            pretrained_model_name_or_path=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            **model_kwargs
        )
