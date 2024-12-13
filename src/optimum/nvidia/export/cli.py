from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from argparse import ArgumentParser


def common_trtllm_export_args(parser: "ArgumentParser"):
    parser.add_argument("model", type=str, help="Model to export.")

    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "--max-input-length",
        type=int,
        default=-1,
        help="Maximum sequence length, in number of tokens, the prompt can be. The maximum number of potential tokens "
        "generated will be <max-output-length> - <max-input-length>.",
    )
    required_group.add_argument(
        "--max-output-length",
        type=int,
        default=-1,
        help="Maximum sequence length, in number of tokens, the model supports.",
    )
    required_group.add_argument(
        "--max-new-tokens", type=int, default=-1, help="Maximum new tokens, "
    )

    multi_gpu_group = parser.add_argument_group("Multi-GPU support arguments")
    multi_gpu_group.add_argument(
        "--tp", type=int, default=1, help="Tensor Parallel degree"
    )
    multi_gpu_group.add_argument(
        "--pp", type=int, default=1, help="Pipeline Parallel degree"
    )

    optional_group = parser.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "-d",
        "--dtype",
        type=str,
        default="auto",
        help="Computational data type used for the model. Default to 'auto' matching model's data type.",
    )
    optional_group.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help="Maximum number of concurrent requests the model can process. Default to 1.",
    )
    optional_group.add_argument(
        "--max-beams-width",
        type=int,
        default=1,
        help='Maximum number of sampling paths ("beam") to evaluate when decoding new a token. Default to 1.',
    )
    optional_group.add_argument(
        "-q", "--quantization", type=str, help="Path to a quantization recipe file."
    )
    optional_group.add_argument(
        "--destination",
        type=str,
        default=None,
        help="Folder where the resulting exported engines will be stored. Default to Hugging Face Hub cache.",
    )
    optional_group.add_argument(
        "--push-to-hub", type=str, help="Repository to push generated engines to."
    )
