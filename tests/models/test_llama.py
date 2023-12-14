from unittest import TestCase

from parameterized import parameterized
from optimum.nvidia.utils.tests import requires_gpu
from optimum.nvidia.models.llama import LLamaForCausalLM as TrtLlamaForCausalLM



class LLamaForCausalLMTestCase(TestCase):

    @requires_gpu
    @parameterized.expand(["float16", "bfloat16"])
    def test_build_engine_7b_with_tp(self, dtype: str):
        model = TrtLlamaForCausalLM.from_pretrained("huggingface/llama-7b", dtype=dtype)
        self.assertIsNotNone(model)