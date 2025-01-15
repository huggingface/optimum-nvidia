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


import pytest
import torch
from transformers import AutoTokenizer
from utils_testing import clean_cached_engines_for_model

from optimum.nvidia import AutoModelForCausalLM, ExportConfig
from optimum.nvidia.export.config import sharded
from optimum.nvidia.utils.nvml import get_device_count


MODEL_TO_TEST = {
    # "google/gemma-2b-it",
    "meta-llama/Llama-2-7b-chat-hf",
    # "mistralai/Mistral-7B-Instruct-v0.2",
    # "meta-llama/Meta-Llama-3-8B",
    # "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

MODEL_KWARGS_MAPS = {"Mixtral-8x7B-Instruct-v0.1": {"tp": 4}}


@pytest.mark.parametrize("model_id", MODEL_TO_TEST)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("pp", [1])
def test_generation(model_id: str, batch_size: int, tp: int, pp: int):
    if get_device_count() < tp * pp:
        pytest.skip("Not enough GPU on the system")

    # TODO: test batched generation as well.
    # TODO: This is flaky depending on the prompt for Mistral / Gemma, maybe see if it is a bug or not.
    prompts = ["Today I am in Paris and I would like to eat crepes."]
    for _ in range(batch_size - 1):
        prompts.append("I knew about a boy who played")

    # Make sure we remove the potentially already built engines.
    clean_cached_engines_for_model(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inp = tokenizer(prompts, padding=False)

    kwargs = {
        "top_k": 1,
        "top_p": 1.0,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "temperature": 1,
    }

    export_config = ExportConfig(
        dtype="float16",
        max_input_len=128,
        max_batch_size=batch_size,
        max_output_len=1024,
    )
    export_config = sharded(export_config, tp, pp)

    trt_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        force_export=True,
        export_config=export_config,
    )

    trt_generated_ids = trt_model.generate(
        inp["input_ids"], num_beams=1, do_sample=False, max_new_tokens=15, **kwargs
    )

    # TODO: left/right padding is not aligned between Transformers and TRT-LLM.
    assert isinstance(trt_generated_ids, torch.Tensor)
    assert trt_generated_ids.shape[0] == batch_size


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
