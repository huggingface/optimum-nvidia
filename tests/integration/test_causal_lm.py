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
from optimum.nvidia.utils.tests.utils import requires_multi_gpu


MAX_TOKENS_DELTA_PERCENT_ATOL = 10.0

MODEL_MAP = {
    "gemma": ["google/gemma-2b-it", "google/gemma-7b-it"],
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "phi": ["microsoft/phi", "microsoft/phi-1.5", "microsoft/phi-2"],
    "phi3": ["microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3-mini-128k-instruct"],
}


@pytest.mark.parametrize("model_type", MODEL_MAP.keys())
@pytest.mark.parametrize("batch_size", [1, 3])
def test_generation(model_type: str, batch_size: int):
    model_ids = (
        [MODEL_MAP[model_type]]
        if isinstance(MODEL_MAP[model_type], str)
        else MODEL_MAP[model_type]
    )

    torch_dtype = torch.float16  # TODO: test fp8, int4, int8, fp32

    # TODO: test batched generation as well.
    # TODO: This is flaky depending on the prompt for Mistral / Gemma, maybe see if it is a bug or not.
    prompts = ["Today I am in Paris and I would like to eat crepes."]
    for _ in range(batch_size - 1):
        prompts.append("I knew about a boy who played")

    max_new_tokens = 15

    for model_id in model_ids:
        # Make sure we remove the potentially already built engines.
        clean_cached_engines_for_model(model_id)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        inp = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")

        torch_model = TransformersAutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
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
            model_id,
            torch_dtype=torch_dtype,
            max_output_length=1000,
            max_batch_size=batch_size,
        )

        trt_generated_ids, _ = trt_model.generate(
            **inp, num_beams=1, do_sample=False, max_new_tokens=max_new_tokens, **kwargs
        )

        # TODO: left/right padding is not aligned between Transformers and TRT-LLM.
        if batch_size == 1:
            assert torch.equal(trt_generated_ids, torch_generated_ids)
        else:
            assert trt_generated_ids.shape == torch_generated_ids.shape

        torch_text = tokenizer.batch_decode(
            torch_generated_ids, skip_special_tokens=True
        )
        trt_text = tokenizer.batch_decode(trt_generated_ids, skip_special_tokens=True)

        assert torch_text == trt_text


@requires_multi_gpu
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
        clean_cached_engines_for_model(model_id)

        pipe_torch = transformers_pipeline(
            task="text-generation",
            model=model_id,
            device="cuda:1",
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

        pipe_trt = pipeline(
            task="text-generation", model=model_id, max_output_length=1000
        )

        with torch.no_grad():
            res_trt = pipe_trt(
                "Today I am in Paris and I would like to eat crepes.",
                max_new_tokens=20,
                **kwargs,
            )

        transformers_output = res_torch[0]["generated_text"]
        trtllm_output = res_trt[0]["generated_text"]

        def count_indexwise_difference(lhs, rhs) -> (int, int):
            maximum_overlapping_span = min(len(transformers_output), len(trtllm_output))

            lhs_ = lhs[:maximum_overlapping_span]
            rhs_ = rhs[:maximum_overlapping_span]
            count = 0

            for l, r in zip(lhs_, rhs_):
                if l != r:
                    count += 1

            return count, maximum_overlapping_span

        num_mismatched_tokens, num_tokens = count_indexwise_difference(
            transformers_output, trtllm_output
        )
        mismatch_percent = float(num_mismatched_tokens) / float(num_tokens) * 100

        assert (
            mismatch_percent <= MAX_TOKENS_DELTA_PERCENT_ATOL
        ), f"{num_mismatched_tokens} mismatched tokens over {num_tokens} > {MAX_TOKENS_DELTA_PERCENT_ATOL} % ({mismatch_percent} %)"
