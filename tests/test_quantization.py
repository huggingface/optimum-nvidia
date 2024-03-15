import pytest
import torch

from optimum.nvidia.quantization import Float8QuantizationConfig
from optimum.nvidia.quantization.ammo.config import KV_CACHE_CFG


@pytest.mark.parametrize(
    "kv_cache,lm_head",
    [
        (False, True),
        (False, True),
    ]
)
def test_float8_ammo_configuration(kv_cache, lm_head):
    config = Float8QuantizationConfig(with_quantized_kv_cache=kv_cache, with_quantized_lm_head=lm_head)

    assert config.requires_calibration
    assert not config.has_calibration_dataset
    assert config.weight_dtype == torch.float8_e4m3fn

    assert config.has_quantized_lm_head == lm_head
    assert config.has_quantized_kv_cache == kv_cache

    qconfig = config.as_ammo_config()
    assert qconfig is not None
    assert "quant_cfg" in qconfig

    quant_cfg = qconfig["quant_cfg"]

    for pattern in {"*weight_quantizer", "*input_quantizer", "default"}:
        assert pattern in quant_cfg
        assert quant_cfg[pattern]["num_bits"] == (4, 3)
        assert quant_cfg[pattern]["axis"] is None

    assert quant_cfg["*lm_head*"]["enable"] == lm_head

    if kv_cache:
        for kv_pattern in KV_CACHE_CFG.keys():
            assert quant_cfg[kv_pattern]["enable"]
            assert quant_cfg[kv_pattern]["axis"] is None
            assert quant_cfg[kv_pattern]["num_bits"] == (4, 3) 
            

