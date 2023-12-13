from typing import Protocol
from unittest import TestCase

from optimum.nvidia.configs import QuantizationConfig
from tensorrt_llm.quantization import QuantMode


class BaseQuantizationConfigTestCase(Protocol):

    def test_has_quantization_step(self):
        raise NotImplementedError()

class NoQuantizationConfigTestCase(BaseQuantizationConfigTestCase, TestCase):
    """
    Test setup for cases without quantization involved
    """

    def setUp(self):
        self.qconfig = QuantizationConfig(QuantMode(0))

    def test_has_quantization_step(self):
        self.assertFalse(self.qconfig.has_quantization_step, "has_quantization_step should be False")


class Float8QuantizationConfigTestCase(BaseQuantizationConfigTestCase, TestCase):
    """
    Test setup for cases for float8 quantization
    """

    def setUp(self):
        self.qconfig = QuantizationConfig(QuantMode.from_description(
            use_fp8_qdq=True,
            use_fp8_kv_cache=True
        ))

    def test_has_quantization_step(self):
        self.assertTrue(self.qconfig.has_quantization_step)