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


# Setup logging needs to happen before importing TRT ...
setup_logging(False)

# ruff: disable=E402
from optimum.nvidia import TensorRTEngineBuilder, TensorRTForCausalLM
from optimum.nvidia.models.llama import LlamaWeightAdapter
from optimum.nvidia.utils.cli import (
    postprocess_quantization_parameters,
    register_common_model_topology_args,
    register_optimization_profiles_args,
    register_quantization_args,
)


LOGGER = getLogger(__name__)


if __name__ == "__main__":
    parser = ArgumentParser("🤗 TensorRT-LLM Llama implementation")
    parser.add_argument("--hub-token", type=str, help="Hugging Face Hub Token to retrieve private weights.")
    register_common_model_topology_args(parser)
    register_optimization_profiles_args(parser)
    register_quantization_args(parser)  # Inject params.quantization_config

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

        login(
            args.hub_token,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # Define the target engine details
    builder = (
        TensorRTEngineBuilder.from_pretrained(args.model, adapter=LlamaWeightAdapter)
        .to(args.dtype)
        .shard(args.tensor_parallelism, args.pipeline_parallelism, args.world_size, args.gpus_per_node)
        .with_generation_profile(args.max_batch_size, args.max_prompt_length, args.max_new_tokens)
        .with_sampling_strategy(args.max_beam_width)
    )

    # Check if we need to collect calibration samples
    if args.has_quantization_step:
        from optimum.nvidia.quantization import get_default_calibration_dataset

        max_length = min(args.max_prompt_length + args.max_new_tokens, tokenizer.model_max_length)
        calib = get_default_calibration_dataset(args.num_calibration_samples)

        if hasattr(calib, "tokenize"):
            calib.tokenize(tokenizer, max_length=max_length, pad_to_multiple_of=8)

        # Add the quantization step
        builder.with_quantization_profile(args.quantization_config, calib)

    # Build the engine
    builder.build(args.output, args.optimization_level)

    with open(args.output.joinpath("build.json"), mode="r", encoding="utf-8") as config_f:
        from json import load

        config = load(config_f)
        model = TensorRTForCausalLM(config, args.output, args.gpus_per_node)

        prompt = "What is the latest generation of Nvidia GPUs?"
        tokens = tokenizer(prompt, padding=True, return_tensors="pt")
        generated, lengths = model.generate(
            **tokens,
            top_k=40,
            top_p=0.95,
            repetition_penalty=10,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
        )

        print(
            tokenizer.decode(
                generated.squeeze().tolist(),
            )
        )

    print(f"TRTLLM engines have been saved at {args.output}.")
