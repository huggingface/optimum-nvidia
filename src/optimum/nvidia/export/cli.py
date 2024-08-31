from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from argparse import ArgumentParser


def common_trtllm_export_args(parser: "ArgumentParser"):
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
