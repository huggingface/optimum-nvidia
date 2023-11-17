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

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", token=True)

    # Define the target engine details
    builder = TRTEngineBuilder.from_pretrained(args.model, adapter=LlamaWeightAdapter) \
        .to(args.dtype) \
        .shard(args.tensor_parallelism, args.pipeline_parallelism, args.world_size, args.gpus_per_node) \
        .with_generation_profile(args.max_batch_size, args.max_prompt_length, args.max_new_tokens) \
        .with_sampling_strategy(args.max_beam_width)

    # Check if we need to collect calibration samples
    if args.has_quantization_step:
        from optimum.nvidia.quantization import HfDatasetCalibration
        calib = HfDatasetCalibration.from_datasets(
            args.dataset,
            split="train",
            num_samples=args.num_calibration_samples,
            column="question",
            streaming=True
        )
        calib.tokenize(tokenizer, tokenizer.model_max_length, pad_to_multiple_of=16)

        # Add the quantization step
        builder.with_quantization_profile(args.quantization_config, calib)

    # Build the engine
    builder.build(args.output)
    
    with open(args.output.joinpath("config.json"), mode="r", encoding="utf-8") as config_f:
        from json import load
        from transformers import AutoTokenizer, pipeline, TextGenerationPipeline

        config = load(config_f)
        tokenizer = AutoTokenizer.from_pretrained(args.model, auth_token=args.hub_token)
        model = TRTEngineForCausalLM(config, args.output, args.gpus_per_node)

        # while True:
        #     prompt = input("Enter text... ")

        prompt = "Who is Nvidia's CEO?"
        tokens = tokenizer(prompt, return_tensors="pt")
        generated = model.generate(
            **tokens,
            top_k=40,
            top_p=0.7,
            repetition_penalty=10,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        print(tokenizer.decode(generated.flatten().tolist(), remove_special_tokens=True))

    print(f"TRTLLM engines have been saved at {args.output}.")
