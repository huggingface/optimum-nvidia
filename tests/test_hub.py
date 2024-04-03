from pathlib import Path
from tempfile import TemporaryDirectory

import mock

# import pytest
import optimum.nvidia.hub
from optimum.nvidia import AutoModelForCausalLM
from optimum.nvidia.hub import FOLDER_TRTLLM_ENGINES
from transformers import AutoConfig as HfAutoConfig
from transformers import AutoModelForCausalLM as HfAutoModelForCausalLM


# from optimum.nvidia.utils.nvml import get_device_name


# def test_load_engine_from_huggingface_hub():
#     with mock.patch("optimum.nvidia.hub.HuggingFaceHubModel.convert_and_build"):
#         device = get_device_name(0)
#
#         try:
#             model = AutoModelForCausalLM.from_pretrained(
#                 "optimum-nvidia/llama-ci", revision=device[-1].lower()
#             )
#             assert model is not None
#             assert (
#                 optimum.nvidia.hub.HuggingFaceHubModel.convert_and_build.call_count == 0
#             )
#         except ValueError:
#             pytest.skip(
#                 f"No revision found for optimum-nvidia/llama-ci on GPU: {device[-1].lower()}"
#             )


def test_save_engine_locally_and_reload():
    with TemporaryDirectory() as out:
        out = Path(out)
        hf_out = out.joinpath("_hf")
        trtllm_out = out.joinpath("_trtllm")

        def _save():
            config = HfAutoConfig.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
            config.num_hidden_layers = 1

            model = HfAutoModelForCausalLM.from_config(config)
            model.save_pretrained(hf_out)

            model = AutoModelForCausalLM.from_pretrained(hf_out)
            model.save_pretrained(trtllm_out)

            assert trtllm_out.exists()
            assert (trtllm_out / FOLDER_TRTLLM_ENGINES).exists()
            assert (trtllm_out / FOLDER_TRTLLM_ENGINES / "rank0.engine").exists()

        def _reload():
            with mock.patch("optimum.nvidia.hub.HuggingFaceHubModel.convert_and_build"):
                model = AutoModelForCausalLM.from_pretrained(trtllm_out)
                assert model is not None
                assert (
                    optimum.nvidia.hub.HuggingFaceHubModel.convert_and_build.call_count
                    == 0
                )

        _save()
        _reload()
