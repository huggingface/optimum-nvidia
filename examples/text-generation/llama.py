#  coding=utf-8
#  Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path

from transformers import AutoTokenizer
from optimum.nvidia import setup_logging

# Setup logging
setup_logging(False)

from optimum.nvidia import TRTEngineBuilder, TRTEngineForCausalLM
from optimum.nvidia.models.llama import LlamaWeightAdapter
from optimum.nvidia.utils.cli import *

LOGGER = getLogger(__name__)


if __name__ == '__main__':
    parser = ArgumentParser("🤗 TensorRT-LLM Llama implementation")
    parser.add_argument("--hub-token", type=str, help="Hugging Face Hub Token to retrieve private weights.")
    register_common_model_topology_args(parser)
    register_optimization_profiles_args(parser)
    register_quantization_args(parser)  # Inject params.quantization_config
    register_triton_server_args(parser)

    parser.add_argument("model", type=str, help="The model's id or path to use.")
    parser.add_argument("output", type=Path, help="Path to store generated TensorRT engine.")
    args = parser.parse_args()
    args = postprocess_quantization_parameters(args)

    # Ensure the output folder exists or create the folder
    if args.output.exists():
        if not args.output.is_dir():
            raise ValueError(f"Output path {args.output} should be an empty folder")

        if any(args.output.iterdir()):
            raise ValueError(f"Output path {args.output} is not empty")
    else:
        LOGGER.info(f"Creating folder {args.output}")
        args.output.mkdir()

    LOGGER.info(f"Exporting {args.model} to TensorRT-LLM engine at {args.output}")
    if args.hub_token is not None:
        from huggingface_hub import login
        login(args.hub_token, )

    if args.has_quantization_step:
        from optimum.nvidia.weights.quantization.ammo import AmmoQuantizer
        LOGGER.info(f"About to calibrate model for quantization {args.quantization_config}")
        quantizer = AmmoQuantizer.from_pretrained(args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=True)
    # builder = TRTEngineBuilder.from_pretrained(args.model, adapter=LlamaWeightAdapter) \
    #     .to(args.dtype) \
    #     .shard(args.tensor_parallelism, args.pipeline_parallelism, args.world_size, args.gpus_per_node) \
    #     .with_quantization_profile(args.quantization_config) \
    #     .with_generation_profile(args.max_batch_size, args.max_prompt_length, args.max_new_tokens) \
    #     .with_sampling_strategy(args.max_beam_width)
    # builder.build(args.output)

    if args.with_triton_structure:
        # generator = TritonLayoutGenerator()
        LOGGER.info(f"Exporting Triton Inference Server structure at {args.output}")
        tokenizer_output = args.output.joinpath("tokenizer/")
        tokenizer.save_pretrained(tokenizer_output)

    with open(args.output.joinpath("config.json"), mode="r", encoding="utf-8") as config_f:
        from json import load
        config = load(config_f)

        with open(args.output.joinpath("llama_float16_tp1_rank0.engine"), mode="rb") as model_f:
            from tensorrt_llm import Mapping
            engine = model_f.read()
            model = TRTEngineForCausalLM(config, Mapping(), engine, use_cuda_graph=False)

    print(f"TRTLLM engines have been saved at {args.output}.")
