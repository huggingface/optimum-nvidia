import pytest

from optimum.nvidia import AutoModelForCausalLM
from optimum.nvidia.compression import AlreadyQuantizedModelException


@pytest.mark.parametrize(
    "model_id",
    [
        "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
    ]
)
def test_prequantized_model_throw(model_id: str):
    with pytest.raises(AlreadyQuantizedModelException):
        _ = AutoModelForCausalLM.from_pretrained(model_id)
