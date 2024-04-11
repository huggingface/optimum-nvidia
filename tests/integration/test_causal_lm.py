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

import gc

import pytest
import torch
from transformers import AutoModelForCausalLM as TransformersAutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import pipeline as transformers_pipeline
from utils_testing import clean_cached_engines_for_model

from optimum.nvidia import AutoModelForCausalLM
from optimum.nvidia.pipelines import pipeline


MODEL_MAP = {
    "gemma": "google/gemma-2b-it",
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
}


@pytest.mark.parametrize("model_type", MODEL_MAP.keys())
def test_generation(model_type: str):
    model_ids = (
        [MODEL_MAP[model_type]]
        if isinstance(MODEL_MAP[model_type], str)
        else MODEL_MAP[model_type]
    )

    torch_dtype = torch.float16  # TODO: test fp8, int4, int8, fp32

    # TODO: test batched generation as well.
    # TODO: This is flaky depending on the prompt for Mistral / Gemma, maybe see if it is a bug or not.
    prompts = ["Today I am in Paris and I would like to eat crepes."]

    max_new_tokens = 40

    for model_id in model_ids:
        # Make sure we remove the potentially already built engines.
        clean_cached_engines_for_model(model_id)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        inp = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")

        torch_model = TransformersAutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, attn_implementation="eager"
        )
        torch_model = torch_model.eval()
        torch_model = torch_model.to("cuda")  # TODO: remove?

        kwargs = {
            "top_k": 1,
            "top_p": 0,
            "length_penalty": 1,
            "repetition_penalty": 1,
            "temperature": 1,
        }
        torch_generated_ids = torch_model.generate(
            **inp, num_beams=1, do_sample=False, max_new_tokens=max_new_tokens, **kwargs
        )

        # Free a bit of memory.
        del torch_model
        gc.collect()
        torch.cuda.empty_cache()

        trt_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype
        )

        trt_generated_ids, _ = trt_model.generate(
            **inp, num_beams=1, do_sample=False, max_new_tokens=max_new_tokens, **kwargs
        )

        assert torch.equal(trt_generated_ids, torch_generated_ids)


@pytest.mark.parametrize("model_type", MODEL_MAP.keys())
def test_pipeline(model_type: str):
    model_ids = (
        [MODEL_MAP[model_type]]
        if isinstance(MODEL_MAP[model_type], str)
        else MODEL_MAP[model_type]
    )

    kwargs = {
        "top_k": 1,
        "top_p": 0,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "temperature": 1,
    }

    for model_id in model_ids:
        # Make sure we remove the potentially already built engines.
        # clean_cached_engines_for_model(model_id)

        pipe_torch = transformers_pipeline(
            task="text-generation",
            model=model_id,
            device="cuda",
            torch_dtype=torch.float16,
        )

        with torch.no_grad():
            res_torch = pipe_torch(
                "Today I am in Paris and I would like to eat crepes.",
                add_special_tokens=True,
                max_new_tokens=20,
                **kwargs,
            )

        # Free a bit of memory.
        del pipe_torch
        gc.collect()
        torch.cuda.empty_cache()

        pipe_trt = pipeline(task="text-generation", model=model_id)

        with torch.no_grad():
            res_trt = pipe_trt(
                "Today I am in Paris and I would like to eat crepes.",
                max_new_tokens=20,
                **kwargs,
            )

        assert res_torch[0]["generated_text"] == res_trt[0]["generated_text"]
