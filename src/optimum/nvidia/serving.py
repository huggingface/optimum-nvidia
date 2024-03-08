import re
from enum import Enum
from logging import getLogger
from types import SimpleNamespace

import requests
from pathlib import Path
from typing import Dict, Union, Optional, Tuple, Any, NamedTuple, List

from tensorrt_llm.bindings import GptJsonConfig

from optimum.nvidia import AutoModelForCausalLM
from optimum.nvidia.utils import rgetattr


LOGGER = getLogger()


OPTIMUM_NVIDIA_TRITON_TEMPLATES_URL = \
    "https://raw.githubusercontent.com/huggingface/optimum-nvidia/inference-endpoint/templates/triton"

_INJECTION_VARIABLE_TEMPLATE_RE = re.compile(r'\${(.*?)}')
_INJECTION_RULES = {
    "VAR_MAX_BATCH_SIZE": "int:self._config.model_config.max_batch_size",
    "VAR_MAX_BEAM_WIDTH": "int:self._config.model_config.max_beam_width",
    "VAR_GPT_MODEL_PATH": "str:paths.llm.logical",
    "VAR_GPT_MODEL_TYPE": "inflight_fused_batching",
    "VAR_BATCH_SCHEDULER_POLICY": "guaranteed_no_evict",  # For now
    "VAR_ENABLE_STREAMING": "bool:true",
    "VAR_ENABLE_OVERLAP": "bool:true",
    "VAR_KV_CACHE_MEMORY_PERCENT": "float:0.9"
}

_AUTHORIZED_MAPPER = {
    "bool": bool,
    "float": float,
    "int": int,
    "str": str,
}


TritonPath = NamedTuple("TritonPath", [("physical", Path), ("logical", Path)])


class ModelTarget(Enum):
    TEXT_GENERATION = "text-generation"
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"
    LLM = "llm"

    @staticmethod
    def values():
        return {
            ModelTarget.TEXT_GENERATION,
            ModelTarget.PREPROCESSING,
            ModelTarget.POSTPROCESSING,
            ModelTarget.LLM,
        }


def _get_content_for(target: ModelTarget) -> Optional[str]:
    url = OPTIMUM_NVIDIA_TRITON_TEMPLATES_URL + "/" + target.value + "/config.pbtxt"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to retrieve the config.pbtxt for target: {target} at {url}")

    return response.content.decode("utf-8")


class TritonLayoutPaths:

    def __init__(self, physical_root: Path, logical_root: Path):
        # Path on the current machine where file will be stored
        self._physical_root = physical_root
        self._logical_root = logical_root

    @property
    def text_generation(self) -> TritonPath:
        return TritonPath(
            self._physical_root.joinpath(ModelTarget.TEXT_GENERATION.value),
            self._logical_root.joinpath(ModelTarget.TEXT_GENERATION.value)
        )

    @property
    def llm(self) -> TritonPath:
        return TritonPath(
            self._physical_root.joinpath(ModelTarget.LLM.value),
            self._logical_root.joinpath(ModelTarget.LLM.value)
        )

    @property
    def preprocessing(self) -> TritonPath:
        return TritonPath(
            self._physical_root.joinpath(ModelTarget.PREPROCESSING.value),
            self._logical_root.joinpath(ModelTarget.PREPROCESSING.value)
        )

    @property
    def postprocessing(self) -> TritonPath:
        return TritonPath(
            self._physical_root.joinpath(ModelTarget.POSTPROCESSING.value),
            self._logical_root.joinpath(ModelTarget.POSTPROCESSING.value)
        )

    def itertargets(self) -> List[Tuple[ModelTarget, TritonPath]]:
        return [
            (ModelTarget.TEXT_GENERATION, self.text_generation),
            (ModelTarget.LLM, self.llm),
            (ModelTarget.PREPROCESSING, self.preprocessing),
            (ModelTarget.POSTPROCESSING, self.postprocessing),
        ]


class TritonLayout:
    def __init__(self, config: GptJsonConfig, ):
        self._config = config

    def _inject_variables(
        self,
        content: str,
        paths: TritonLayoutPaths,
        rules: Dict[str, str] = None,
        scope: Dict[str, Any] = None
    ) -> (str, int):
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

            # Mapper is more of a sanity checker for str -> target -> str conversion
            value = mapper(value)

            # Replace
            content = content.replace(variable.group(0), str(value), 1)
            injected += 1

        # Number of variable injected
        return content, injected

    @staticmethod
    def from_pretrained(model_id_or_path: Union[str, Path]) -> 'TritonLayout':
        model = AutoModelForCausalLM.from_pretrained(model_id_or_path)
        return TritonLayout(model.config)

    @staticmethod
    def from_config_file(config_path: Path):
        return TritonLayout(GptJsonConfig.parse_file(config_path))

    def save_pretrained(self, root: Path, only_config: bool = False, overrides: Dict[str, str] = None):
        if not root.exists():
            root.mkdir(parents=True)

        if not overrides:
            overrides = {}

        # Paths can either be physical path on the machine or
        # a logical path mapping another layout (in a container for instance)
        physical_root = logical_root = root
        if "root" in overrides:
            logical_root = Path(overrides["root"])

        paths = TritonLayoutPaths(physical_root, logical_root)
        for (target, triton_path) in paths.itertargets():
            LOGGER.debug(f"Injecting content for target {target}")

            content = _get_content_for(target)
            content_injected, num_injected_vars = self._inject_variables(content, paths)

            LOGGER.debug(f"Injected {num_injected_vars} variables for target {target}")

            triton_path.physical.mkdir(parents=True, exist_ok=True)
            with open(triton_path.physical / "config.pbtxt", "w", encoding="utf-8") as config_f:
                LOGGER.debug(f"Writing config.pbtxt for target {target} at: {config_f.name}")
                config_f.write(content_injected)

            (triton_path.physical / "1").mkdir(exist_ok=True)

            if not only_config:
                raise NotImplementedError()

    def push_to_hub(self):
        raise NotImplementedError()


class TritonInferenceEndpoint(TritonLayout):
    def save_pretrained(self, root: Path, only_config: bool = False):
        super().save_pretrained(root, only_config, {"root": "/opt/endpoint"})
