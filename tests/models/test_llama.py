from typing import NamedTuple

import pytest

from optimum.nvidia.models.llama import LLamaForCausalLM as TrtLlamaForCausalLM
from optimum.nvidia.utils.tests import requires_gpu


LlamaModelInfo = NamedTuple(
    "LlamaModelInfo",
    [
        ("model_id", str),
        ("dtype", "str"),
        ("tp_degree", int),
        ("pp_degre", int),
        ("world_size", int)
    ]
)


@pytest.fixture(scope="module")
def llama(request) -> TrtLlamaForCausalLM:
    info: LlamaModelInfo = request.param
    yield TrtLlamaForCausalLM.from_pretrained(info.model_id, dtype=info.dtype)


@pytest.mark.parametrize(
    "llama",
    [
        LlamaModelInfo("huggingface/llama-7b", "float16", 1, 1, 1),
        LlamaModelInfo("huggingface/llama-7b", "bfloat16", 1, 1, 1)
    ],
    ids=[
        "huggingface/llama-7b (float16)",
        "huggingface/llama-7b (bfloat16)",
    ],
    indirect=True
)
@requires_gpu
def test_build_llama(llama: TrtLlamaForCausalLM):
    assert llama is not None
    assert llama.config.name == "llama"



