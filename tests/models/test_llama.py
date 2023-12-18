from typing import NamedTuple

import pytest

from optimum.nvidia.lang import DataType
from optimum.nvidia.models.llama import LLamaForCausalLM as TrtLlamaForCausalLM
from optimum.nvidia.utils.tests import requires_gpu


LlamaModelInfo = NamedTuple(
    "LlamaModelInfo", [("model_id", str), ("dtype", "str"), ("tp_degree", int), ("pp_degre", int), ("world_size", int)]
)


@pytest.fixture(scope="module")
def llama(request) -> TrtLlamaForCausalLM:
    info = request.param
    yield TrtLlamaForCausalLM.from_pretrained(info.model_id, dtype=info.dtype)


@pytest.mark.parametrize(
    "llama, dtype",
    [
        (LlamaModelInfo("huggingface/llama-7b", "float16", 1, 1, 1), "float16"),
         (LlamaModelInfo("huggingface/llama-7b", "bfloat16", 1, 1, 1), "bfloat16"),
    ],
    ids=[
        "huggingface/llama-7b (float16)",
        "huggingface/llama-7b (bfloat16)",
    ],
    indirect=["llama"],
)
@requires_gpu
def test_build_llama(llama: TrtLlamaForCausalLM, dtype: str):
    assert llama is not None
    assert llama.config.name == "llama"
    assert llama.config.precision == dtype, f"Precision should be {dtype} but got {llama.config.precision}"

    model_config = llama.config
    assert model_config.data_type == DataType(dtype).as_trt()
    assert model_config.use_gpt_attention_plugin, f"GPT Attention plugin should be enabled"
    assert model_config.use_packed_input, f"Remove Padding should set to true with GPT Attention plugin"
