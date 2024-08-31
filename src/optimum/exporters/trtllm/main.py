from argparse import ArgumentParser

from transformers import AutoConfig

from optimum.nvidia import AutoModelForCausalLM
from optimum.nvidia.export import ExportConfig
from optimum.nvidia.export.cli import common_trtllm_export_args


if __name__ == "__main__":
    parser = ArgumentParser("Hugging Face Optimum TensorRT-LLM exporter")
    common_trtllm_export_args(parser)
    parser.add_argument(
        "--push-to-hub",
        type=str,
        required=False,
        help="Repository id where to push engines",
    )
    parser.add_argument(
        "destination", help="Local path where the generated engines will be saved"
    )
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model)
    export = ExportConfig.from_config(config, args.max_batch_size)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, export_config=export, export_only=True
    )
    model.save_pretrained(args.destination)

    if args.push_to_hub:
        print(f"Exporting model to the Hugging Face Hub: {args.push_to_hub}")
        model.push_to_hub(
            args.push_to_hub,
            commit_message=f"Optimum-CLI TensorRT-LLM {args.model} export",
        )
