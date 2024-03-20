import tempfile

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import AutoModelForCausalLM as HfAutoModelForCausalLM

from optimum.nvidia import AutoModelForCausalLM
from optimum.nvidia.quantization import AutoQuantizationConfig
from optimum.nvidia.utils.tests.utils import requires_float8


@pytest.mark.parametrize(
    "model_id",
    ["google/gemma-2b", "meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1"],
)
@requires_float8()
def test_float8_causallm_use_fp8(model_id: str):
    # Use a tinier model
    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = 1

    # Create the flow and convert
    with tempfile.TemporaryDirectory() as tmp_f:
        _ = AutoTokenizer.from_pretrained(model_id).save_pretrained(tmp_f)
        _ = HfAutoModelForCausalLM.from_config(config).save_pretrained(tmp_f)
        model = AutoModelForCausalLM.from_pretrained(
            tmp_f, max_batch_size=1, beam_width=1, use_fp8=True
        )

        assert model is not None


@pytest.mark.parametrize(
    "model_id",
    # ["google/gemma-2b", "meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1"],
    ["meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1"],
)
@pytest.mark.parametrize("weight", ["float8", torch.float8_e4m3fn])
@pytest.mark.parametrize("activation", ["float8", torch.float8_e4m3fn])
@pytest.mark.parametrize("dataset", ["c4-new"])
@requires_float8()
def test_float8_causallm_custom_qconfig_predefined_dataset(
    model_id: str, dataset: str, weight, activation
):
    # Use a tiner model
    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = 1

    # Create the flow and convert
    with tempfile.TemporaryDirectory() as tmp_f:
        _ = HfAutoModelForCausalLM.from_config(config).save_pretrained(tmp_f)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        qconfig = AutoQuantizationConfig.from_description(
            weight=weight,
            activation=activation,
            tokenizer=tokenizer,
            dataset=dataset,
            num_samples=32,
            max_sequence_length=128,
        )
        model = AutoModelForCausalLM.from_pretrained(
            tmp_f, max_batch_size=1, beam_width=1, quantization_config=qconfig
        )
        assert model is not None
