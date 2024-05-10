#  coding=utf-8
#  Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from argparse import ArgumentParser, Namespace

from tensorrt_llm.quantization import QuantMode


# Model topology (sharding, pipelining, dtype)
def register_common_model_topology_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--tensor-parallelism",
        type=int,
        default=1,
        dest="tp",
        help="Define the number of slice for each tensor each GPU will receive.",
    )
    parser.add_argument(
        "--pipeline-parallelism",
        type=int,
        default=1,
        dest="pp",
        help="Define the number of sections to split neural network layers",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Total number of GPUs over all the nodes.",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=1,
        help="Total number of GPUs on a single node.",
    )
    parser.add_argument(
        "-o",
        "--opt-level",
        type=int,
        default=2,
        dest="optimization_level",
        help="Optimization level between 0 (no optimization) and 5 (maximum level of optimization).",
    )
    return parser


# TensorRT's optimization profiles
def register_optimization_profiles_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help="Maximum batch size for the model.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=128,
        help="Maximum prompt a user can give.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--max-beam-width",
        type=int,
        default=1,
        help="Maximum number of beams for sampling",
    )

    return parser


# Triton Inference Server
def register_triton_server_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--with-triton-structure",
        action="store_true",
        help="Generate the Triton Inference Server structure",
    )
    return parser


# Quantization
def register_quantization_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--fp8", action="store_true", help="Enable FP8 quantization for Ada & Hopper."
    )
    parser.add_argument(
        "--fp8-cache",
        action="store_true",
        help="Enable KV cache as fp8 for Ada & Hopper.",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=512,
        help="How much samples to use when calibrating.",
    )
    return parser


def postprocess_quantization_parameters(params: Namespace) -> Namespace:
    # Only support FP8 quantization for now
    qconfig = QuantMode.from_description(
        quantize_weights=False,
        quantize_activations=False,
        per_token=False,
        per_channel=False,
        per_group=False,
        use_int4_weights=False,
        use_int8_kv_cache=False,
        use_fp8_kv_cache=params.fp8_cache,
        use_fp8_qdq=params.fp8,
    )

    params.has_quantization_step = qconfig != QuantMode(0)
    params.quantization_config = qconfig

    # If we do have the output path, then let's create the calibration path
    if "output" in params:
        from pathlib import Path

        params.calibration_output = Path(params.output).joinpath("calibration")

    return params
