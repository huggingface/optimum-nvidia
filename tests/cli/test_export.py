from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from integration.utils_testing import clean_cached_engines_for_model
from tensorrt_llm.bindings import GptJsonConfig

from optimum.nvidia.export import Workspace
from optimum.nvidia.utils.nvml import get_device_name


if TYPE_CHECKING:
    from pytest_console_scripts import ScriptRunner


def _ensure_required_folder_and_files_exists(root: Workspace):
    assert root.checkpoints_path.exists()
    assert (root.checkpoints_path / "config.json").exists()
    assert (root.checkpoints_path / "rank0.safetensors").exists()

    assert root.engines_path.exists()
    assert (root.engines_path / "config.json").exists()
    assert (root.engines_path / "rank0.engine").exists()


def test_optimum_export_default(runner: "ScriptRunner") -> None:
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device_id = get_device_name(0)[-1]

    default_dest = Workspace.from_hub_cache(model_id, device_id)
    out = runner.run(f"optimum-cli export trtllm {model_id}")
    assert out.success

    _ensure_required_folder_and_files_exists(default_dest)

    exported_config = GptJsonConfig.parse_file(
        default_dest.engines_path / "config.json"
    )
    assert exported_config.model_config.max_batch_size == 1
    assert exported_config.model_config.max_beam_width == 1
    assert exported_config.model_config.max_input_len >= 1

    clean_cached_engines_for_model(model_id)


def test_optimum_export_custom_destination(runner: "ScriptRunner") -> None:
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    with TemporaryDirectory() as dest:
        default_dest = Workspace(Path(dest))
        out = runner.run(
            f"optimum-cli export trtllm --destination {default_dest.root} {model_id}"
        )
        assert out.success

        _ensure_required_folder_and_files_exists(default_dest)
        exported_config = GptJsonConfig.parse_file(
            default_dest.engines_path / "config.json"
        )
        assert exported_config.model_config.max_batch_size == 1
        assert exported_config.model_config.max_beam_width == 1
        assert exported_config.model_config.max_input_len >= 1
