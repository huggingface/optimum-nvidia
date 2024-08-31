import subprocess
import sys
from typing import TYPE_CHECKING, Optional

from ...nvidia.export.cli import common_trtllm_export_args
from ..base import BaseOptimumCLICommand, CommandInfo


if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace, _SubParsersAction


class TrtLlmExportCommand(BaseOptimumCLICommand):
    COMMAND = CommandInfo(
        name="trtllm", help="Export PyTorch models to TensorRT-LLM compiled engines"
    )

    def __init__(
        self,
        subparsers: "_SubParsersAction",
        args: Optional["Namespace"] = None,
        command: Optional["CommandInfo"] = None,
        from_defaults_factory: bool = False,
        parser: Optional["ArgumentParser"] = None,
    ):
        super().__init__(
            subparsers,
            args=args,
            command=command,
            from_defaults_factory=from_defaults_factory,
            parser=parser,
        )
        self.args_string = " ".join(sys.argv[3:])

    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        return common_trtllm_export_args(parser)

    def run(self):
        full_command = f"python3 -m optimum.exporters.trtllm {self.args_string}"
        subprocess.run(full_command, shell=True, check=True)
