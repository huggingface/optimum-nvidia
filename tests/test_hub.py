from pathlib import Path
from tempfile import TemporaryDirectory

import mock
import pytest
from huggingface_hub.utils import RevisionNotFoundError

import optimum.nvidia.hub
from optimum.nvidia import AutoModelForCausalLM
from optimum.nvidia.hub import FOLDER_TRTLLM_ENGINES
from optimum.nvidia.utils.nvml import get_device_name


def test_load_engine_from_huggingface_hub():
    with mock.patch("optimum.nvidia.hub.HuggingFaceHubModel.convert_and_build"):
        device_id = get_device_name(0).lower()

        try:
            model = AutoModelForCausalLM.from_pretrained(
                "optimum-nvidia/llama-ci", revision=device_id
            )
            assert model is not None
            assert (
                optimum.nvidia.hub.HuggingFaceHubModel.convert_and_build.call_count == 0
            )
        except RevisionNotFoundError:
            pytest.skip(
                f"No revision found for optimum-nvidia/llama-ci on GPU: {device_id}"
            )


def test_save_engine_locally_and_reload():
    with TemporaryDirectory() as out:

        def _save():
            model = AutoModelForCausalLM.from_pretrained("optimum-nvidia/llama-ci")
            model.save_pretrained(out)

            outpath = Path(out)
            assert outpath.exists()
            assert (outpath / FOLDER_TRTLLM_ENGINES).exists()
            assert (outpath / FOLDER_TRTLLM_ENGINES / "rank0.engine").exists()

        def _reload():
            with mock.patch("optimum.nvidia.hub.HuggingFaceHubModel.convert_and_build"):
                model = AutoModelForCausalLM.from_pretrained(out)
                assert model is not None
                assert (
                    optimum.nvidia.hub.HuggingFaceHubModel.convert_and_build.call_count
                    == 0
                )

        _save()
        _reload()
