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
from typing import Optional

import pytest
import torch
from utils_testing import clean_cached_engines_for_model

from optimum.nvidia import AutoModelForCausalLM
from transformers import AutoModelForCausalLM as TransformersAutoModelForCausalLM
from transformers import AutoTokenizer


MODEL_MAP = {
    #"gemma": ["google/gemma-2b", "google/gemma-7b"],
    "gemma": ["google/gemma-2b"],
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
}

@pytest.mark.parametrize("model_type", MODEL_MAP.keys())
@pytest.mark.parametrize("max_new_tokens", [None])  # TODO: add a test
def test_generation(model_type: str, max_new_tokens: Optional[int]):
    model_ids = [MODEL_MAP[model_type]] if isinstance(MODEL_MAP[model_type], str) else MODEL_MAP[model_type]

    torch_dtype = torch.float16  # TODO: test fp8, int4, int8, fp32

    #prompts = ["Today I am in Paris and", "I am", "I would like"]
    prompts = ["Today I am in Paris and"]

    kwargs = {}
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = max_new_tokens

    for model_id in model_ids:
        # Make sure we remove the potentially already built engines.
        clean_cached_engines_for_model(model_id)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        inp = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")

        with torch.device("cuda"):
            torch_model = TransformersAutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch_dtype, attn_implementation="eager"
            )
        torch_model = torch_model.eval()
        torch_model = torch_model.to("cuda")  # TODO: remove?

        torch_generated_ids = torch_model.generate(**inp, num_beams=1, do_sample=False, top_k=None, **kwargs)

        # Free a bit of memory.
        del torch_model
        gc.collect()
        torch.cuda.empty_cache()

        trt_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)

        trt_generated_ids = trt_model.generate(**inp, num_beams=1, do_sample=False, top_k=None, **kwargs)

        print("trt_generated_ids", trt_generated_ids.shape, trt_generated_ids)
        print("trt decode", tokenizer.batch_decode(trt_generated_ids))

        print("torch_generated_ids", torch_generated_ids.shape, torch_generated_ids)
        print("torch decode", tokenizer.batch_decode(torch_generated_ids))
        assert torch.equal(trt_generated_ids, torch_generated_ids)
