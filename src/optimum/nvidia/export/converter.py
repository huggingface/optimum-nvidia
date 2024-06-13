from abc import ABC
from enum import Enum
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import Optional, Sequence, Type, Union

from tensorrt_llm import BuildConfig, Mapping
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
        models: Union[PretrainedModel, Sequence[PretrainedModel]],
        mapping: Optional[Mapping] = None,
    ) -> TensorRTArtifact:
        """
        Take a local model and create the intermediate TRTLLM checkpoint
        :param models
        :param mapping
        :return:
        """
        if isinstance(models, PretrainedModel):
            models = [models]

        for rank, model in enumerate(models):
            LOGGER.info(
                f"Converting {models[0].config.architecture} model for rank {rank} to TRTLLM"
            )
            model.save_checkpoint(str(self._workspace.checkpoints_path))

        return TensorRTArtifact.checkpoints(str(self._workspace.checkpoints_path))

    def build(
        self,
        models: Union[PretrainedModel, Sequence[PretrainedModel]],
        config: BuildConfig,
    ) -> TensorRTArtifact:
        """
        :param models
        :param config
        :return:
        """
        if isinstance(models, PretrainedModel):
            models = [models]

        for rank, model in enumerate(models):
            LOGGER.info(f"Building TRTLLM engine for rank {rank}")

            engine = build(model, config)
            engine.save(str(self._workspace.engines_path))

        return TensorRTArtifact.engines(str(self._workspace.engines_path))
