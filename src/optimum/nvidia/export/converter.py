from abc import ABC
from enum import Enum
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import Optional, Union, Type

from tensorrt_llm import Mapping, BuildConfig
from tensorrt_llm.builder import build
from tensorrt_llm.models import PretrainedModel

from optimum.nvidia.export import Workspace

LOGGER = getLogger()


class TensorRTArtifactKind(Enum):
    CHECKPOINTS = "checkpoints"
    ENGINES = "engines"


class TensorRTArtifact:

    @staticmethod
    def checkpoints(root: Union[str, PathLike]) -> "TensorRTArtifact":
        return TensorRTArtifact(TensorRTArtifactKind.CHECKPOINTS, root)

    @staticmethod
    def engines(root: Union[str, PathLike]) -> "TensorRTArtifact":
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

    def __init__(self, workspace: Optional[Union[Workspace, str, bytes, Path]] = None):
        if not workspace:
            workspace = Workspace.from_hub_cache()

        if isinstance(workspace, (str, bytes, Path)):
            workspace = Workspace(Path(workspace))

        self._workspace = workspace

    def convert(
        self,
        model: PretrainedModel,
        mapping: Optional[Mapping] = None
    ) -> TensorRTArtifact:
        """
        Take a local model and create the intermediate TRTLLM checkpoint
        :param local_path_to_model
        :param mapping
        :return:
        """
        LOGGER.info(f"Converting {model.config.architecture} model to TRTLLM")
        model.save_checkpoint(str(self._workspace.checkpoints_path))
        return TensorRTArtifact.checkpoints(str(self._workspace.checkpoints_path))

    def build(self, model: PretrainedModel, config: BuildConfig) -> TensorRTArtifact:
        """
        :param model
        :param config
        :return:
        """
        LOGGER.info(f"Building TRTLLM engine from checkpoint {self._workspace.checkpoints_path}")

        engine = build(model, config)
        engine.save(str(self._workspace.engines_path))

        return TensorRTArtifact.engines(str(self._workspace.engines_path))