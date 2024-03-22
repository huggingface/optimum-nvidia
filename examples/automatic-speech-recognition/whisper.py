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

from huggingface_hub import login

from optimum.nvidia import setup_logging


# Setup logging needs to happen before importing TRT ...
setup_logging(False)

from optimum.nvidia.models.whisper import WhisperForConditionalGeneration
from optimum.nvidia.utils.cli import (
    register_common_model_topology_args,
    register_optimization_profiles_args,
    register_quantization_args,
)


LOGGER = getLogger(__name__)

if __name__ == "__main__":
    parser = ArgumentParser("🤗 TensorRT-LLM Whisper implementation")
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

    if args.hub_token is not None:
        login(args.hub_token)

    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.save_pretrained(args.output)
    print(f"TRTLLM engines have been saved at {args.output}.")
