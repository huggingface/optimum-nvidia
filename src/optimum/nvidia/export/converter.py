from abc import ABC
from enum import Enum
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Type, Union

from tensorrt_llm.builder import build

from optimum.nvidia.export import Workspace
from optimum.nvidia.utils.nvml import get_device_name, is_post_ampere


if TYPE_CHECKING:
    from tensorrt_llm import BuildConfig, Mapping
    from tensorrt_llm.models import PretrainedModel

LOGGER = getLogger()


def infer_plugin_from_build_config(config: "BuildConfig") -> "BuildConfig":
    if is_post_ampere():
        # Required for Chunk Context
        LOGGER.debug("Enabling Paged Context FMHA plugin")
        config.plugin_config.update_from_dict({"use_paged_context_fmha": True})

    return config


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

    def __init__(
        self,
        model_id: str,
        subpart: str = "",
        workspace: Optional[Union["Workspace", str, bytes, Path]] = None,
    ):
        LOGGER.info(f"Creating a model converter for {subpart}")
        if not workspace:
            target_device = get_device_name(0)[-1]
            workspace = Workspace.from_hub_cache(
                model_id, target_device, subpart=subpart
            )

        if isinstance(workspace, (str, bytes, Path)):
            workspace = Workspace(Path(workspace))

        LOGGER.debug(f"Initializing model converter workspace at {workspace.root}")

        self._workspace = workspace

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    def quantize(self):
        raise NotImplementedError()

    def convert(
        self,
        models: Union["PretrainedModel", Sequence["PretrainedModel"]],
        mapping: Optional["Mapping"] = None,
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
        models: Union["PretrainedModel", Sequence["PretrainedModel"]],
        config: "BuildConfig",
    ) -> TensorRTArtifact:
        """
        :param models
        :param config
        :return:
        """

        if not isinstance(models, Sequence):
            models = [models]

        config = infer_plugin_from_build_config(config)

        for rank, model in enumerate(models):
            LOGGER.info(f"Building TRTLLM engine for rank {rank}")

            engine = build(model, config)
            engine.save(str(self._workspace.engines_path))

        return TensorRTArtifact.engines(str(self._workspace.engines_path))
