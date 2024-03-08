from pathlib import Path
from typing import Optional, TYPE_CHECKING

from optimum.commands import BaseOptimumCLICommand, CommandInfo
from optimum.nvidia.serving import TritonInferenceEndpoint

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace, _SubParsersAction


def parse_triton_args(parser: ArgumentParser):
    parser.add_argument(
        "--for-hf-endpoint",
        action="store_true",
        help="Generate Triton layout to deploy on HF Inference Endpoint"
    )

    parser.add_argument("model_path", type=str, help="Local path where the model can be found")


class TritonPackageCommand(BaseOptimumCLICommand):
    COMMAND = CommandInfo(
        name="triton",
        help="Package a TensorRT-LLM model to Triton Inference Server compatible layout."
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
            subparsers, args=args, command=command, from_defaults_factory=from_defaults_factory, parser=parser
        )

    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        return parse_triton_args(parser)

    def run(self):
        args = self.args

        args.model_path = Path(args.model_path)
        if not args.model_path.exists():
            raise ValueError(f"Path {args.model_path} doesn't exist")

        if not (engine_path := (args.model_path / "engines")).exists():
            raise ValueError(f"Path {engine_path} doesn't exist. Did you export your model to TensorRT-LLM?")

        if not (engine_config_path := (engine_path / "config.json")).exists():
            raise ValueError(f"Cannot find config.json in {engine_path}.")

        layout = TritonInferenceEndpoint.from_config_file(engine_config_path)
        layout.save_pretrained("/opt/endpoint")
