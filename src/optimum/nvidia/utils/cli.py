from argparse import ArgumentParser

from optimum.nvidia.lang import DataType


def register_common_model_topology_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--dtype",
        choices=[dtype.value for dtype in DataType], default=DataType.FLOAT16,
        help="Data type to do the computations."
    )
    parser.add_argument(
        "--tensor-parallelism",
        type=int, default=1,
        help="Define the number of slice for each tensor each GPU will receive."
    )
    parser.add_argument(
        "--pipeline-parallelism",
        type=int, default=1,
        help="Define the number of sections to split neural network layers"
    )
    parser.add_argument("--world-size", type=int, default=1, help="Total number of GPUs over all the nodes.")
    parser.add_argument("--gpus-per-node", type=int, default=1, help="Total number of GPUs on a single node.")
    return parser


def register_optimization_profiles_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--max-batch-size", type=int, default=1, help="Maximum batch size for the model.")
    parser.add_argument("--max-prompt-length", type=int, default=128, help="Maximum prompt a user can give.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--max-beam-width", type=int, default=1, help="Maximum number of beams for sampling")

    return parser


def register_triton_server_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--with-triton-structure", action="store_true",
        help="Generate the Triton Inference Server structure"
    )
    return parser