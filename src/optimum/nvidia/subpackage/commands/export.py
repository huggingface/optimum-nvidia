import sys
from typing import TYPE_CHECKING, Optional, Union

from transformers import AutoConfig, AutoTokenizer

from optimum.commands import optimum_cli_subcommand
from optimum.commands.base import BaseOptimumCLICommand, CommandInfo
from optimum.commands.export.base import ExportCommand
from optimum.nvidia import AutoModelForCausalLM, ExportConfig
from optimum.nvidia.export.cli import common_trtllm_export_args


if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace, _SubParsersAction
    from pathlib import Path


OPTIMUM_NVIDIA_CLI_QUANTIZATION_TARGET_REF = "TARGET_QUANTIZATION_RECIPE"


def import_source_file(fname: Union[str, "Path"], modname: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(modname, fname)
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)


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

        # Do we have quantization?
        if args.quantization:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            import_source_file(args.quantization, "recipe")

            try:
                from recipe import TARGET_QUANTIZATION_RECIPE

                qconfig = TARGET_QUANTIZATION_RECIPE(tokenizer)
            except ImportError:
                raise ModuleNotFoundError(
                    f"Global variable 'TARGET_QUANTIZATION_RECIPE' was not found in {args.quantization}. "
                    "This is required to automatically detect and allocate the right recipe for quantization."
                )

        else:
            qconfig = None

        # Allocate model and derivatives needed to export
        config = AutoConfig.from_pretrained(args.model)
        export = ExportConfig.from_config(config, args.max_batch_size)

        if args.max_input_length > 0:
            export.max_input_len = args.max_input_length

        if args.max_output_length > 0:
            export.max_output_len = args.max_output_length

        if args.max_new_tokens > 0:
            export.max_num_tokens = args.max_new_tokens

        # Import sharding
        export = export.with_sharding(args.tp, args.pp)

        # Export
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            export_config=export,
            quantization_config=qconfig,
            export_only=True,
            force_export=True,
        )

        if args.destination:
            model.save_pretrained(args.destination)

        if args.push_to_hub:
            print(f"Exporting model to the Hugging Face Hub: {args.push_to_hub}")
            model.push_to_hub(
                args.push_to_hub,
                commit_message=f"Optimum-CLI TensorRT-LLM {args.model} export",
            )
