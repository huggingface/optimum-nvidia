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

from pathlib import Path
from json import load
import torch
from transformers import AutoProcessor
from datasets import load_dataset

from optimum.nvidia import setup_logging

from huggingface_hub import login

from logging import getLogger

# Setup logging needs to happen before importing TRT ...
setup_logging(False)
LOGGER = getLogger(__name__)

from optimum.nvidia import TensorRTForSpeechSeq2SeqEngineBuilder, TensorRTForSpeechSeq2Seq
from optimum.nvidia.utils.cli import (
    postprocess_quantization_parameters,
    register_common_model_topology_args,
    register_optimization_profiles_args,
    register_quantization_args,
)


if __name__ == "__main__":
    parser = ArgumentParser("🤗 TensorRT-LLM Whisper implementation")
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
        login(
            args.hub_token,
        )

    # Define the target engine details
    builder = (
        TensorRTForSpeechSeq2SeqEngineBuilder.from_pretrained(args.model)
        .to(args.dtype)
        .shard(args.tensor_parallelism, args.pipeline_parallelism, args.world_size, args.gpus_per_node)
        .with_generation_profile(args.max_batch_size)
        .with_sampling_strategy(args.max_beam_width)
    )

    # Check if we need to collect calibration samples
    if args.has_quantization_step:
        raise NotImplementedError("TODO")

    # Build the engine
    builder.build(args.output, args.optimization_level)

    print(f"TRTLLM engines have been saved at {args.output}.")

    """
    with open(args.output.joinpath("build.json"), mode="r", encoding="utf-8") as config_f:
        config = load(config_f)
        model = TensorRTForSpeechSeq2SeqLMLM(config, args.output, args.gpus_per_node)

        processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        input_features = inputs.input_features



        generated_ids = model.generate(inputs=input_features)

        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        transcription

        generated, lengths = model.generate(
            input_features,
            top_k=40,
            top_p=0.95,
            repetition_penalty=10,
            max_new_tokens=args.max_new_tokens,
        )

        generated_text = tokenizer.decode(generated[0, 0])
        print(generated_text)
    """
