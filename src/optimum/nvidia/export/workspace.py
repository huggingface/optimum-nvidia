from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import cached_assets_path
from tensorrt_llm import __version__ as TRTLLM_VERSION

from optimum.nvidia import LIBRARY_NAME
from optimum.nvidia.export import (
    PATH_FILE_CHECKPOINTS,
    PATH_FILE_ENGINES,
    PATH_FOLDER_CHECKPOINTS,
    PATH_FOLDER_ENGINES,
)
from optimum.nvidia.utils.nvml import get_device_name


@dataclass
class Workspace:
    root: Path

    @staticmethod
    def from_hub_cache(
        namespace: str = LIBRARY_NAME, version: str = TRTLLM_VERSION
    ) -> "Workspace":
        device_name = get_device_name(0)[-1]
        return Workspace(
            cached_assets_path(namespace, namespace=version, subfolder=device_name)
        )

    def __post_init__(self):
        if not self.checkpoints_path.exists():
            self.checkpoints_path.mkdir(parents=True)

        if not self.engines_path.exists():
            self.engines_path.mkdir(parents=True)

    @property
    def checkpoints_path(self) -> Path:
        """
        Folder path location holding all the engines
        :return: `Path`
        """
        return self.root / PATH_FOLDER_CHECKPOINTS

    @property
    def engines_path(self) -> Path:
        """
        Folder path location holding all the engines
        :return: `Path`
        """
        return self.root / PATH_FOLDER_ENGINES

    @property
    def checkpoints(self) -> Iterable[Path]:
        """
        Generator discovering all the checkpoint files present in this workspace
        :return:
        """
        return self.checkpoints_path.glob(PATH_FILE_CHECKPOINTS)

    def engines(self) -> Iterable[Path]:
        """
        Generator discovering all the engine files present in this workspace
        :return:
        """
        return self.engines_path.glob(PATH_FILE_ENGINES)
