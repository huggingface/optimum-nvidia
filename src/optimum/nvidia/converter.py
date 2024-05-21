from abc import ABC
from enum import Enum
from logging import getLogger
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union, Dict, Any, Type

from tensorrt_llm import Mapping
from tensorrt_llm.models import PretrainedConfig

from optimum.nvidia.builder import LocalEngineBuilder
from transformers import AutoConfig, PretrainedConfig as TransformersPretrainedConfig

LOGGER = getLogger()


class TensorRTArtifactKind(Enum):
    CHECKPOINTS = "checkpoints"
    ENGINES = "engines"


class TensorRTArtifact:

    @staticmethod
    def checkpoints(root: Union[str, PathLike]) -> "TensorRTArtifact":
        return TensorRTArtifact(TensorRTArtifactKind.CHECKPOINTS, root)

    @staticmethod
    def engines(root: Union[str, PathLike]):
        return TensorRTArtifact(TensorRTArtifactKind.ENGINES, root)

    def __init__(self, kind: TensorRTArtifactKind, root: Union[str, PathLike]):
        self._kind = kind
        self._root = root

    @property
    def kind(self) -> TensorRTArtifactKind:
        return self._kind

    @property
    def root(self) -> Path:
        return Path(self._root)

    def push_to_hub(self):
        raise NotImplementedError()


class TensorRTModelConverter(ABC):
    CONFIG_CLASS: Type
    MODEL_CLASS: Type

    def __init__(self, dest: Optional[Union[str, PathLike]] = None, workspace: Optional[Union[str, PathLike]] = None):
        self._dest = dest
        self._workspace = workspace or TemporaryDirectory()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        LOGGER.info(f"Saving ")

        if isinstance(self._workspace, TemporaryDirectory):
            LOGGER.debug(f"Cleaning temporary folder {self._workspace.name}")
            self._workspace.cleanup()

    def convert(self, local_weights_path: Union[str, PathLike], config: PretrainedConfig, mapping: Mapping) -> TensorRTArtifact:
        """
        Take a locally downloaded model from Hugging Face and create the intermediate TRTLLM checkpoint
        :param local_weights_path:
        :param config:
        :param mapping: Compute parallelization information for the model conversion (TP/PP/MOE)
        :return:
        """
        LOGGER.info(f"Converting Hugging Face checkpoint {local_weights_path} to TRTLLM")


        return None

    def build(self) -> TensorRTArtifact:
        """
        Takes in
        :return:
        """
        LOGGER.info(f"Building TRTLLM engine from checkpoint {self._workspace.name}")
        # engine_builder = LocalEngineBuilder(model_config, engines_folder)
        # engine_builder.build(engine_config)
        #
        # return [engines_folder], [engines_folder.relative_to(local_path)]
        return None