import subprocess
import sys
from typing import TYPE_CHECKING, Optional

from ..base import BaseOptimumCLICommand, CommandInfo


if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace, _SubParsersAction


def parse_args_trtllm(parser: "ArgumentParser"):
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Model ID on huggingface.co or path on disk to load model from.",
    )
    required_group.add_argument(
        "--max-input-length",
        type=int,
        default=1,
        help="Maximum sequence length, in number of tokens, the prompt can be. The maximum number of potential tokens "
        "generated will be <max-output-length> - <max-input-length>.",
    )
    required_group.add_argument(
        "--max-output-length",
        type=int,
        default=1,
        help="Maximum sequence length, in number of tokens, the model supports.",
    )

    optional_group = parser.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "-d",
        "--dtype",
        type=str,
        default="auto",
        help="Computational data type used for the model.",
    )
    optional_group.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help="Maximum number of concurrent requests the model can process.",
    )
    optional_group.add_argument(
        "--max-beams-width",
        type=int,
        default=1,
        help='Maximum number of sampling paths ("beam") to evaluate when decoding new a token.',
    )


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
        return parse_args_trtllm(parser)

    def run(self):
        full_command = f"python3 -m optimum.exporters.neuron {self.args_string}"
        subprocess.run(full_command, shell=True, check=True)
