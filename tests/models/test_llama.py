from parameterized import parameterized

from optimum.nvidia.models.llama import LLamaForCausalLM as TrtLlamaForCausalLM
from optimum.nvidia.utils.tests import requires_gpu


@parameterized.expand(["float16", "bfloat16"])
@requires_gpu
def test_build_engine_7b_with_tp(dtype: str):
    model = TrtLlamaForCausalLM.from_pretrained("huggingface/llama-7b", dtype=dtype)
    assert model
