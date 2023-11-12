from argparse import ArgumentParser, Namespace

from optimum.nvidia.lang import DataType
from optimum.nvidia.configs import QuantizationConfig

from tensorrt_llm.quantization import QuantMode


# Model topology (sharding, pipelining, dtype)
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


# TensorRT optimization profiles
def register_optimization_profiles_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--max-batch-size", type=int, default=1, help="Maximum batch size for the model.")
    parser.add_argument("--max-prompt-length", type=int, default=128, help="Maximum prompt a user can give.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--max-beam-width", type=int, default=1, help="Maximum number of beams for sampling")

    return parser


# Triton Inference Server
def register_triton_server_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--with-triton-structure", action="store_true",
        help="Generate the Triton Inference Server structure"
    )
    return parser


# Quantization
def register_quantization_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--fp8", action="store_true", help="Enable FP8 quantization for Ada & Hopper.")
    parser.add_argument("--fp8-cache", action="store_true", help="Enable KV cache as fp8 for Ada & Hopper.")
    return parser


def postprocess_quantization_parameters(params: Namespace) -> Namespace:
    # Only support FP8 quantization for now
    params.quantization_config = QuantizationConfig(
        mode=QuantMode.from_description(
            quantize_weights=False,
            quantize_activations=False,
            per_token=False,
            per_channel=False,
            per_group=False,
            use_int4_weights=False,
            use_int8_kv_cache=False,
            use_fp8_kv_cache=args.fp8_cache,
            use_fp8_qdq=args.fp8
        ),
        group_size=-1
    )

    return params