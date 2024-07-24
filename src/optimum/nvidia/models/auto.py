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
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, Union

from huggingface_hub import ModelHubMixin

from optimum.nvidia.errors import UnsupportedModelException
from optimum.nvidia.models.gemma import GemmaForCausalLM
from optimum.nvidia.models.llama import LlamaForCausalLM
from optimum.nvidia.utils import model_type_from_known_config


if TYPE_CHECKING:
    from optimum.nvidia.export import ExportConfig
    from optimum.nvidia.runtime import CausalLM


class AutoModelForCausalLM(ModelHubMixin):
    """ """

    _SUPPORTED_MODEL_CLASS = {
        "llama": LlamaForCausalLM,
        "mistral": LlamaForCausalLM,
        "mixtral": LlamaForCausalLM,
        "gemma": GemmaForCausalLM,
        # "phi": PhiForCausalLM
    }

    def __init__(self):
        super().__init__()

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
        config: Optional[Dict[str, Any]] = None,
        export_config: Optional["ExportConfig"] = None,
        force_export: bool = False,
        use_cuda_graph: bool = False,
        **model_kwargs,
    ) -> "CausalLM":
        if config is None:
            raise ValueError("Unable to determine the model type with config = None")

        model_type = model_type_from_known_config(config)

        if (
            not model_type
            or model_type not in AutoModelForCausalLM._SUPPORTED_MODEL_CLASS
        ):
            raise UnsupportedModelException(model_type)

        model_clazz = AutoModelForCausalLM._SUPPORTED_MODEL_CLASS[model_type]
        model = model_clazz.from_pretrained(
            pretrained_model_name_or_path=model_id,
            config=config,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            export_config=export_config,
            force_export=force_export,
            use_cuda_graph=use_cuda_graph,
            **model_kwargs,
        )

        return model
