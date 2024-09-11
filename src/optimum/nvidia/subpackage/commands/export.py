import sys
from typing import TYPE_CHECKING, Optional

from transformers import AutoConfig

from optimum.commands import optimum_cli_subcommand
from optimum.commands.base import BaseOptimumCLICommand, CommandInfo
from optimum.commands.export.base import ExportCommand
from optimum.nvidia import AutoModelForCausalLM, ExportConfig
from optimum.nvidia.export.cli import common_trtllm_export_args


if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace, _SubParsersAction


@optimum_cli_subcommand(ExportCommand)
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
        # Retrieve args from CLI
        args = self.args

        # Allocate model and derivatives needed to export
        config = AutoConfig.from_pretrained(args.model)
        export = ExportConfig.from_config(config, args.max_batch_size)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, export_config=export, export_only=True
        )

        if args.destination:
            model.save_pretrained(args.destination)

        if args.push_to_hub:
            print(f"Exporting model to the Hugging Face Hub: {args.push_to_hub}")
            model.push_to_hub(
                args.push_to_hub,
                commit_message=f"Optimum-CLI TensorRT-LLM {args.model} export",
            )
