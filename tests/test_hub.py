from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

import mock
import pytest
import torch.cuda
from transformers import AutoConfig as HfAutoConfig
from transformers import AutoModelForCausalLM as HfAutoModelForCausalLM

import optimum.nvidia.hub
from optimum.nvidia import AutoModelForCausalLM, ExportConfig
from optimum.nvidia.export import Workspace
from optimum.nvidia.export.config import sharded
from optimum.nvidia.hub import folder_list_checkpoints, folder_list_engines
from optimum.nvidia.utils import model_type_from_known_config
from optimum.nvidia.utils.nvml import get_device_name


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
    [
        ("meta-llama/Llama-2-7b-chat-hf", 1),
        ("google/gemma-2b", 1),
        ("mistralai/Mistral-7B-v0.3", 4),
    ],
)
def test_save_engine_locally_and_reload(model_id: Tuple[str, int]):
    with TemporaryDirectory() as hf_out:
        with TemporaryDirectory() as trtllm_out:
            device_name = get_device_name(0)[-1]
            trtllm_out = Path(trtllm_out)

            def _save():
                config = HfAutoConfig.from_pretrained(model_id[0])
                config.num_hidden_layers = 1

                model = HfAutoModelForCausalLM.from_config(config)
                model.save_pretrained(hf_out)
                del model
                torch.cuda.empty_cache()

                export_config = ExportConfig(
                    dtype="float16",
                    max_input_len=128,
                    max_batch_size=1,
                    max_output_len=128,
                    max_num_tokens=100,
                )
                export_config = sharded(export_config, model_id[1], 1)

                model = AutoModelForCausalLM.from_pretrained(
                    hf_out, export_config=export_config
                )
                model.save_pretrained(trtllm_out)
                del model
                torch.cuda.empty_cache()

                assert trtllm_out.exists()
                assert (trtllm_out / device_name / "engines" / "config.json").exists()
                assert (
                    trtllm_out / device_name / "engines" / "generation_config.json"
                ).exists()
                assert (trtllm_out / device_name / "engines" / "rank0.engine").exists()

            def _reload():
                with mock.patch("optimum.nvidia.export.TensorRTModelConverter.build"):
                    workspace = Workspace(trtllm_out / device_name)
                    model = AutoModelForCausalLM.from_pretrained(workspace.engines_path)
                    assert model is not None
                    assert (
                        optimum.nvidia.export.TensorRTModelConverter.build.call_count
                        == 0
                    )

                    del model
                    torch.cuda.empty_cache()

            _save()
            _reload()


@pytest.mark.parametrize(
    "type",
    (
        ("llama", "LlamaForCausalLM"),
        ("gemma", "GemmaForCausalLM"),
        ("mistral", "MistralForCausalLM"),
        ("mixtral", "MixtralForCausalLM"),
    ),
)
def test_model_type_from_known_config(type: Tuple[str, str]):
    transformers_type, trtllm_type = type

    # transformers config
    transformers_config = {"model_type": transformers_type}
    assert model_type_from_known_config(transformers_config) == transformers_type

    # trtllm engine config
    tensorrt_llm_config = {"pretrained_config": {"architecture": trtllm_type}}
    assert model_type_from_known_config(tensorrt_llm_config) == transformers_type


def test_model_type_from_known_config_fail():
    assert model_type_from_known_config({"": ""}) is None

    with pytest.raises(RuntimeError):
        model_type_from_known_config(
            {"pretrained_config": {"architecture": "_LlamaForCausaLM"}}
        )

    with pytest.raises(RuntimeError):
        model_type_from_known_config(
            {"pretrained_config": {"architecture": "_llamaForCausaLM"}}
        )

    with pytest.raises(RuntimeError):
        model_type_from_known_config(
            {"pretrained_config": {"architecture": "llama_ForCausaLM"}}
        )

    with pytest.raises(RuntimeError):
        model_type_from_known_config(
            {"pretrained_config": {"architecture": "_lLamaForCausaLM"}}
        )

    with pytest.raises(RuntimeError):
        model_type_from_known_config(
            {"pretrained_config": {"architecture": "llamaforcausalm"}}
        )

    with pytest.raises(RuntimeError):
        model_type_from_known_config(
            {"pretrained_config": {"architecture": "123llamaforcausalm"}}
        )
        model_type_from_known_config(
            {"pretrained_config": {"architecture": "llama123forcausalm"}}
        )
