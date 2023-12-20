from typing import NamedTuple, Tuple

import pytest
from tensorrt_llm.bindings import DataType as TrtDataType
from transformers import PreTrainedTokenizer, AutoTokenizer, TensorType

from optimum.nvidia.lang import DataType
from optimum.nvidia.models.llama import LlamaForCausalLM as TrtLlamaForCausalLM
from optimum.nvidia.utils.tests import requires_gpu


LlamaModelInfo = NamedTuple(
    "LlamaModelInfo", [
        ("model_id", str),
        ("dtype", str),
        ("tp_degree", int),
        ("pp_degree", int),
        ("world_size", int)
    ]
)


@pytest.fixture(scope="module")
def llama(request) -> Tuple[PreTrainedTokenizer, TrtLlamaForCausalLM]:
    info = request.param

    if not isinstance(info, LlamaModelInfo):
        info, _ = info

    tokenizer = AutoTokenizer.from_pretrained(info.model_id, padding_side="left")
    model = TrtLlamaForCausalLM.from_pretrained(info.model_id, dtype=info.dtype, max_batch_size=2)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


@pytest.mark.parametrize(
    "llama, expected_precision",
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
def test_build(llama: Tuple[PreTrainedTokenizer, TrtLlamaForCausalLM], expected_precision: str):
    assert llama is not None

    tokenizer, llama = llama
    assert llama.config.name == "llama"
    assert llama.config.precision == expected_precision, \
        f"Precision should be {expected_precision} but got {llama.config.precision}"

    model_config = llama.config.model_config
    assert model_config.data_type == TrtDataType(DataType(dtype).as_trt())
    assert model_config.use_gpt_attention_plugin, "GPT Attention plugin should be enabled"
    assert model_config.use_packed_input, "Remove Padding should set to true with GPT Attention plugin"


@pytest.mark.parametrize(
    "llama",
    [
        LlamaModelInfo("huggingface/llama-7b", "float16", 1, 1, 1),
        LlamaModelInfo("huggingface/llama-7b", "bfloat16", 1, 1, 1),
    ],
    ids=[
        "huggingface/llama-7b (float16)",
        "huggingface/llama-7b (bfloat16)",
    ],
    indirect=["llama"],
)
@pytest.mark.parametrize("inputs, num_samples", [
    ("This is an example", 1),
    (["This is an example"], 1),
    (["This is an example", "And this is a second example"], 2)
])
@requires_gpu
def test_run_simple(llama: Tuple[PreTrainedTokenizer, TrtLlamaForCausalLM], inputs: str, num_samples: int):
    tokenizer, llama = llama

    assert llama is not None
    encodings = tokenizer(inputs, padding=True, return_tensors=TensorType.PYTORCH)
    generated, lengths = llama.generate(**encodings)

    assert generated.shape[0] == num_samples
