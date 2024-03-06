import re
from enum import Enum
from logging import getLogger
from types import SimpleNamespace

import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union, Optional, Tuple

from optimum.nvidia import AutoModelForCausalLM
from optimum.nvidia.runtime import CausalLM
from optimum.nvidia.utils import rgetattr


LOGGER = getLogger()


OPTIMUM_NVIDIA_TRITON_TEMPLATES_URL = \
    "https://raw.githubusercontent.com/huggingface/optimum-nvidia/inference-endpoint/templates/triton"

_AUTHORIZED_MAPPER = {"bool": bool, "float": float, "int": int, "str": str}
_INJECTION_VARIABLE_TEMPLATE_RE = re.compile(r'\${(.*?)}')
_INJECTION_RULES = {
    "VAR_MAX_BATCH_SIZE": "int:self._model.config.model_config.max_batch_size",
    "VAR_MAX_BEAM_WIDTH": "int:self._model.config.model_config.max_beam_width",
    "VAR_TOKENIZER_DIR": "str:paths.tokenizer",
    "VAR_GPT_MODEL_PATH": "str:paths.llm",
    "VAR_GPT_MODEL_TYPE": "inflight_fused_batching",
    "VAR_BATCH_SCHEDULER_POLICY": "guaranteed_no_evict",  # For now
    "VAR_ENABLE_STREAMING": "bool:true",
    "VAR_ENABLE_OVERLAP": "bool:true",
    "VAR_KV_CACHE_MEMORY_PERCENT": "float:0.9"
}



class ModelTarget(Enum):
    TEXT_GENERATION = "text-generation"
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"
    LLM = "llm"
    TOKENIZER = "tokenizer"


def _get_content_for(target: ModelTarget) -> Optional[str]:
    if target == ModelTarget.TOKENIZER:
        return None  # No content to fetch here

    url = OPTIMUM_NVIDIA_TRITON_TEMPLATES_URL + "/" + target.value + "/config.pbtxt"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to retrieve the config.pbtxt for target: {target} at {url}")

    return response.content.decode("utf-8")


@dataclass
class TritonLayoutPaths:
    root: Path
    ensemble: Path
    llm: Path
    preprocessing: Path
    postprocessing: Path
    tokenizer: Path

    def __init__(self, root: Path):
        self.root = root
        self.ensemble = root.joinpath(ModelTarget.TEXT_GENERATION.value)
        self.llm = root.joinpath(ModelTarget.LLM.value)
        self.preprocessing = root.joinpath(ModelTarget.PREPROCESSING.value)
        self.postprocessing = root.joinpath(ModelTarget.POSTPROCESSING.value)
        self.tokenizer = root.joinpath(ModelTarget.TOKENIZER.value)

    def itertargets(self) -> Tuple[Tuple[ModelTarget, Path], ...]:
        return (
            (ModelTarget.TEXT_GENERATION, self.ensemble),
            (ModelTarget.LLM, self.llm),
            (ModelTarget.PREPROCESSING, self.preprocessing),
            (ModelTarget.POSTPROCESSING, self.postprocessing),
            (ModelTarget.TOKENIZER, self.tokenizer),
        )


class TritonLayout:
    def __init__(self, model: CausalLM):
        self._model = model

    def _inject_variables(self, content: str, paths: TritonLayoutPaths, rules: Dict[str, str] = None, scope = None) -> (str, int):
        if rules is None:
            rules = _INJECTION_RULES

        if scope is None:
            scope = SimpleNamespace(**locals())

        injected = 0
        while variable := _INJECTION_VARIABLE_TEMPLATE_RE.search(content):
            var_name = variable.group(1)
            if var_name not in rules:
                raise ValueError(f"Unknown variable {var_name}")

            LOGGER.debug(f"\t - Found variable: {var_name}")

            # Retrieve the variable to inject
            target = rules[var_name]
            if ":" in target:
                mapper, target = target.split(":")
                if mapper not in _AUTHORIZED_MAPPER:
                    raise ValueError(f"Unknown mapper function {mapper} for {target}")
                mapper = _AUTHORIZED_MAPPER[mapper]
            else:
                mapper = lambda x: x

            if "." in target and not target.replace(".", "", 1).isdigit():
                value = rgetattr(scope, target)
            else:
                value = target

            print(f"\t - Found variable: {var_name} - {value} @ {mapper}")

            value = mapper(value)

            # Replace
            content = content.replace(variable.group(0), str(value), 1)
            injected += 1

        # Number of variable injected
        return content, injected

    @staticmethod
    def from_pretrained(model_id_or_path: Union[str, Path]) -> 'TritonLayout':
        model = AutoModelForCausalLM.from_pretrained(model_id_or_path)
        return TritonLayout(model)

    def save_pretrained(self, root: Path, only_config: bool = False):
        if not root.exists():
            root.mkdir(parents=True)

        paths = TritonLayoutPaths(root)

        for (target, path) in paths.itertargets():
            LOGGER.debug(f"Injecting content for target {target}")

            content = _get_content_for(target)

            if not target == ModelTarget.TOKENIZER:
                content_injected, num_injected_vars = self._inject_variables(content, paths)

                LOGGER.debug(f"Injected {num_injected_vars} variables for target {target}")

                path.mkdir(parents=True, exist_ok=True)
                with open(path / "config.pbtxt", "w", encoding="utf-8") as config_f:
                    LOGGER.debug(f"Writing config.pbtxt for target {target} at: {config_f.name}")
                    config_f.write(content_injected)

                (path / "1").mkdir(exist_ok=True)

            if not only_config:
                raise NotImplementedError()

    def push_to_hub(self):
        raise NotImplementedError()