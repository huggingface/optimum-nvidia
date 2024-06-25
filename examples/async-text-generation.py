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
import asyncio
from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path

from transformers import AutoTokenizer

from optimum.nvidia import AutoModelForCausalLM, ExportConfig, setup_logging


# Setup logging needs to happen before importing TRT ...
setup_logging(True)

from optimum.nvidia.utils.cli import (
    postprocess_quantization_parameters,
    register_common_model_topology_args,
    register_optimization_profiles_args,
    register_quantization_args,
)


LOGGER = getLogger(__name__)


async def infer():
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    export = ExportConfig.from_pretrained(args.model)
    export.max_input_len = 1024
    export.max_output_len = 256
    export.max_num_tokens = 256
    export.max_beam_width = 1

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", export_config=export
    )
    # model.save_pretrained(args.output)

    prompt = "What is the latest generation of Nvidia GPUs?"
    tokens = tokenizer(prompt, return_tensors="pt")
    generated = await model.agenerate(
        tokens["input_ids"],
    )

    generated_text = tokenizer.batch_decode(generated, skip_special_tokens=True)
    print(generated_text)


if __name__ == "__main__":
    parser = ArgumentParser("🤗 Optimum-Nvidia Text-Generation Example")
    parser.add_argument(
        "--hub-token",
        type=str,
        help="Hugging Face Hub Token to retrieve private weights.",
    )
    register_common_model_topology_args(parser)
    register_optimization_profiles_args(parser)
    register_quantization_args(parser)  # Inject params.quantization_config

    parser.add_argument("model", type=str, help="The model's id or path to use.")
    parser.add_argument(
        "output", type=Path, help="Path to store generated TensorRT engine."
    )
    args = parser.parse_args()
    args = postprocess_quantization_parameters(args)

    if args.hub_token is not None:
        from huggingface_hub import login

        login(args.hub_token)

    asyncio.run(infer())
