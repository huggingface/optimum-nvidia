import shutil
from abc import ABC
from enum import Enum
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Type, Union

from tensorrt_llm.builder import build

from optimum.nvidia.compression.modelopt import ModelOptQuantizer
from optimum.nvidia.export import Workspace
from optimum.nvidia.utils.nvml import get_device_name, is_post_ampere


if TYPE_CHECKING:
    from tensorrt_llm import BuildConfig, Mapping
    from tensorrt_llm.models import PretrainedModel
    from transformers import PreTrainedModel as TransformersPreTrainedModel

    from optimum.nvidia.compression.modelopt import ModelOptRecipe

LOGGER = getLogger()


def infer_plugin_from_build_config(config: "BuildConfig") -> "BuildConfig":
    if is_post_ampere():
        # Required for Chunk Context
        LOGGER.debug("Enabling Paged Context FMHA plugin")
        config.plugin_config.update_from_dict({"use_paged_context_fmha": True})

    config.plugin_config.update_from_dict({"enable_xqa": False})
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
        license_path: Optional[Union[str, bytes, Path]] = None,
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
        self._license_path = license_path

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    def save_license(self, licence_filename: str = "LICENSE"):
        """
        Save the license if provided and if the license is not already present.
        This method doesn't check the content of the license
        :param licence_filename: Name of the file containing the license content
        """
        if (
            not (
                dst_licence_file_path := self.workspace.root / licence_filename
            ).exists()
            and self._license_path
        ):
            shutil.copyfile(self._license_path, dst_licence_file_path)

    def quantize(
        self, model: "TransformersPreTrainedModel", qconfig: "ModelOptRecipe"
    ) -> TensorRTArtifact:
        quantizer = ModelOptQuantizer(qconfig)
        quantizer.preprocess_model(model, workspace=self.workspace)
        quantizer.postprocess_model(model, workspace=self.workspace)

        self.save_license()
        return TensorRTArtifact.checkpoints(self._workspace.checkpoints_path)

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

        self.save_license()
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

        for model in models:
            LOGGER.info(
                f"Building TRTLLM engine for rank {model.config.mapping.rank} ->> {config.to_dict()}"
            )

            engine = build(model, config)
            engine.save(str(self._workspace.engines_path))

        self.save_license()
        return TensorRTArtifact.engines(str(self._workspace.engines_path))
