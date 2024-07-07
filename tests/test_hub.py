from pathlib import Path
from tempfile import TemporaryDirectory

import mock
import pytest
from transformers import AutoConfig as HfAutoConfig
from transformers import AutoModelForCausalLM as HfAutoModelForCausalLM

import optimum.nvidia.hub
from optimum.nvidia import AutoModelForCausalLM


@pytest.mark.parametrize(
    "model_id",
    ("meta-llama/Llama-2-7b-chat-hf", "google/gemma-2b", "mistralai/Mistral-7B-v0.3"),
)
def test_save_engine_locally_and_reload(model_id: str):
    with TemporaryDirectory() as hf_out:
        with TemporaryDirectory() as trtllm_out:
            trtllm_out = Path(trtllm_out)

            def _save():
                config = HfAutoConfig.from_pretrained(model_id)
                config.num_hidden_layers = 1

                model = HfAutoModelForCausalLM.from_config(config)
                model.save_pretrained(hf_out)

                model = AutoModelForCausalLM.from_pretrained(hf_out)
                model.save_pretrained(trtllm_out)

                assert trtllm_out.exists()
                assert (trtllm_out / "rank0.engine").exists()

            def _reload():
                with mock.patch("optimum.nvidia.export.TensorRTModelConverter.build"):
                    model = AutoModelForCausalLM.from_pretrained(trtllm_out)
                    assert model is not None
                    assert (
                        optimum.nvidia.export.TensorRTModelConverter.build.call_count
                        == 0
                    )

            _save()
            _reload()
