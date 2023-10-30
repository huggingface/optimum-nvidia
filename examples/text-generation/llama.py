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
setup_logging(True)

from optimum.nvidia import TRTEngineBuilder
from optimum.nvidia.lang import DataType
from optimum.nvidia.models.llama import LlamaWeightAdapter

LOGGER = getLogger(__name__)


if __name__ == '__main__':
    parser = ArgumentParser("🤗 TensorRT-LLM Llama implementation")
    parser.add_argument(
        "--dtype",
        choices=[dtype.value for dtype in DataType], default=DataType.FLOAT16,
        help="Data type to do the computations."
    )

    # Optimization profiles
    parser.add_argument("--max-batch-size", type=int, default=1, help="Maximum batch size for the model.")
    parser.add_argument("--max-prompt-length", type=int, default=128, help="Maximum prompt a user can give.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--max-beam-width", type=int, default=1, help="Maximum number of beams for sampling")

    # Triton Inference Server related
    parser.add_argument(
        "--with-triton-structure", action="store_true",
        help="Generate the Triton Inference Server structure"
    )
    parser.add_argument("model", type=str, help="The model's id or path to use.")
    parser.add_argument("output", type=Path, help="Path to store generated TensorRT engine.")
    args = parser.parse_args()

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
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    engine = TRTEngineBuilder.from_pretrained(args.model, adapter=LlamaWeightAdapter) \
        .to(args.dtype) \
        .with_generation_profile(args.max_batch_size, args.max_prompt_length, args.max_new_tokens) \
        .with_sampling_strategy(args.max_beam_width) \
        .build(args.output)

    if args.with_triton_structure:
        # generator = TritonLayoutGenerator()
        LOGGER.info(f"Exporting Triton Inference Server structure at {args.output}")
        tokenizer_output = args.output.joinpath("tokenizer/")
        tokenizer.save_pretrained(tokenizer_output)

    print(f"TRTLLM engines have been saved at {args.output}.")
