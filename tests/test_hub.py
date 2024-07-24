from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

import mock
import pytest
from transformers import AutoConfig as HfAutoConfig
from transformers import AutoModelForCausalLM as HfAutoModelForCausalLM

import optimum.nvidia.hub
from optimum.nvidia import AutoModelForCausalLM
from optimum.nvidia.hub import folder_list_checkpoints, folder_list_engines
from optimum.nvidia.utils import model_type_from_known_config


def create_fake_checkpoints_and_engines(root: str, rank: int):
    root = Path(root)
    for i in range(rank):
        with open(root.joinpath(f"rank{i}.safetensors"), "wb") as f:
            f.write(b"abcd")

        with open(root.joinpath(f"rank{i}.engine"), "wb") as f:
            f.write(b"abcd")


@pytest.mark.parametrize("rank", [1, 4])
def test_folder_list_checkpoints(rank: int):
    with TemporaryDirectory() as tmp:
        create_fake_checkpoints_and_engines(tmp, rank)

        checkpoints = list(folder_list_checkpoints(Path(tmp)))
        assert len(checkpoints) == rank, "Wrong number of checkpoints returned"
        assert all(
            (
                lambda checkpoint: checkpoint.name.endswith("safetensors")
                and checkpoint.startswith("rank"),
                checkpoints,
            )
        ), "Invalid checkpoint name detected in the output"


@pytest.mark.parametrize("rank", [1, 4])
def test_folder_list_engines(rank: int):
    with TemporaryDirectory() as tmp:
        create_fake_checkpoints_and_engines(tmp, rank)

        engines = list(folder_list_engines(Path(tmp)))
        assert len(engines) == rank, "Wrong number of engines returned"
        assert all(
            (
                lambda engine: engine.name.endswith("engine")
                and engine.startswith("rank"),
                engines,
            )
        ), "Invalid engine name detected in the output"


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


@pytest.mark.parametrize(
    "type",
    (
        ("llama", "LlamaForCausalLM"),
        ("gemma", "GemmaForCausalLM"),
        ("mistral", "MistralForCausalLM"),
        ("mixtral", "MixtralForCausalLM"),
    )
)
def test_model_type_from_known_config(type: Tuple[str, str]):
    transformers_type, trtllm_type = type

    # transformers config
    transformers_config = {"model_type": transformers_type}
    assert model_type_from_known_config(transformers_config) == transformers_type

    # trtllm engine config
    tensorrt_llm_config = {"pretrained_config": {"architecture": trtllm_type}}
    assert model_type_from_known_config(tensorrt_llm_config) == trtllm_type


def test_model_type_from_known_config_fail():
    assert model_type_from_known_config({"": ""}) is None

    with pytest.raises(RuntimeError):
        model_type_from_known_config({"pretrained_config": {"architecture": "_LlamaForCausaLM"}})
        model_type_from_known_config({"pretrained_config": {"architecture": "Llama_ForCausaLM"}})
        model_type_from_known_config({"pretrained_config": {"architecture": "_llamaForCausaLM"}})
        model_type_from_known_config({"pretrained_config": {"architecture": "llama_ForCausaLM"}})
        model_type_from_known_config({"pretrained_config": {"architecture": "_lLamaForCausaLM"}})
        model_type_from_known_config({"pretrained_config": {"architecture": "llamaforcausalm"}})
        model_type_from_known_config({"pretrained_config": {"architecture": "123llamaforcausalm"}})
        model_type_from_known_config({"pretrained_config": {"architecture": "llama123forcausalm"}})
