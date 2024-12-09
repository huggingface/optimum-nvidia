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
from utils_testing import clean_cached_engines_for_model

from optimum.nvidia import AutoModelForCausalLM, ExportConfig
from optimum.nvidia.export.config import sharded
from optimum.nvidia.utils.nvml import get_device_count
from optimum.nvidia.utils.tests import (
    assert_generated_partially_match,
)


MODEL_TO_TEST = {
    "google/gemma-2b-it",
    "meta-llama/Llama-2-7b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Meta-Llama-3-8B",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

MODEL_KWARGS_MAPS = {"Mixtral-8x7B-Instruct-v0.1": {"tp": 4}}


@pytest.mark.parametrize("model_id", MODEL_TO_TEST)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("pp", [1])
def test_generation(model_id: str, batch_size: int, tp: int, pp: int):
    if get_device_count() < tp * pp:
        pytest.skip("Not enough GPU on the system")

    torch_dtype = torch.float16  # TODO: test fp8, int4, int8, fp32

    # TODO: test batched generation as well.
    # TODO: This is flaky depending on the prompt for Mistral / Gemma, maybe see if it is a bug or not.
    prompts = ["Today I am in Paris and I would like to eat crepes."]
    for _ in range(batch_size - 1):
        prompts.append("I knew about a boy who played")

    max_new_tokens = 15

    # Make sure we remove the potentially already built engines.
    clean_cached_engines_for_model(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inp = tokenizer(prompts, padding=False, return_tensors="pt").to("cuda")

    torch_model = TransformersAutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
        device_map="auto",
    )
    torch_model = torch_model.eval()

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

    export_config = ExportConfig(
        dtype="float16",
        max_input_len=1024,
        max_batch_size=batch_size,
        max_output_len=1000,
        max_num_tokens=max_new_tokens,
    )
    export_config = sharded(export_config, tp, pp)

    trt_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        force_export=True,
        export_config=export_config,
    )

    trt_generated_ids = trt_model.generate(
        inp, num_beams=1, do_sample=False, max_new_tokens=max_new_tokens, **kwargs
    )

    # TODO: left/right padding is not aligned between Transformers and TRT-LLM.
    assert isinstance(trt_generated_ids, torch.tensor)
    assert trt_generated_ids.shape == torch_generated_ids.shape
    for i in range(batch_size):
        mask = inp["attention_mask"][i]
        shift = len(mask) - mask.sum()
        assert_generated_partially_match(
            trt_generated_ids[i].cpu().numpy(),
            torch_generated_ids[i, shift:].cpu().numpy(),
            0.05,
        )


# @requires_multi_gpu
# @pytest.mark.parametrize("model_id", MODEL_TO_TEST)
# def test_pipeline(model_id: str):
#     kwargs = {
#         "top_k": 1,
#         "top_p": 0,
#         "length_penalty": 1,
#         "repetition_penalty": 1,
#         "temperature": 1,
#     }
#
#     # Make sure we remove the potentially already built engines.
#     clean_cached_engines_for_model(model_id)
#
#     pipe_torch = transformers_pipeline(
#         task="text-generation",
#         model=model_id,
#         device="cpu",
#         torch_dtype=torch.float16,
#     )
#
#     with torch.no_grad():
#         res_torch = pipe_torch(
#             "Today I am in Paris and I would like to eat crepes.",
#             add_special_tokens=True,
#             max_new_tokens=20,
#             **kwargs,
#         )
#
#     # Free a bit of memory.
#     del pipe_torch
#     gc.collect()
#     torch.cuda.empty_cache()
#
#     pipe_trt = pipeline(
#         task="text-generation",
#         model=model_id,
#         max_output_length=1000,
#         **MODEL_KWARGS_MAPS.get(model_id, {}),
#     )
#
#     with torch.no_grad():
#         res_trt = pipe_trt(
#             "Today I am in Paris and I would like to eat crepes.",
#             max_new_tokens=20,
#             **kwargs,
#         )
#
#     assert_generated_text_partially_match(
#         res_torch[0]["generated_text"], res_trt[0]["generated_text"], atol=0.05
#     )
