import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Optional, Union

import pytest
from tensorrt_llm.bindings import GptJsonConfig, QuantMode

from optimum.nvidia.export import Workspace
from optimum.nvidia.utils.nvml import get_device_name


if TYPE_CHECKING:
    from pytest_console_scripts import ScriptRunner


def _ensure_required_folder_and_files_exists(
    root: Union[Path, Workspace], device_name: Optional[str] = None
):
    if isinstance(root, Path):
        engines_path = root.joinpath(device_name)
    else:
        engines_path = root.engines_path

    assert engines_path.exists()
    assert (engines_path / "config.json").exists()
    assert (engines_path / "generation_config.json").exists()
    assert (engines_path / "rank0.engine").exists()


def test_optimum_export_default(script_runner: "ScriptRunner") -> None:
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device_id = get_device_name(0)[-1]

    default_dest = Workspace.from_hub_cache(model_id, device_id)
    out = script_runner.run(f"optimum-cli export trtllm {model_id}", shell=True)
    assert out.success

    _ensure_required_folder_and_files_exists(default_dest)

    exported_config = GptJsonConfig.parse_file(
        default_dest.engines_path / "config.json"
    )
    assert exported_config.model_config.max_batch_size == 1
    assert exported_config.model_config.max_beam_width == 1
    assert exported_config.model_config.max_input_len >= 1

    # Clean up
    shutil.rmtree(default_dest.root)


def test_optimum_export_custom_destination(script_runner: "ScriptRunner") -> None:
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device_name = get_device_name(0)[-1]

    with TemporaryDirectory() as dest:
        default_dest = Path(dest)
        out = script_runner.run(
            f"optimum-cli export trtllm --destination {default_dest} {model_id}",
            shell=True,
        )
        assert out.success

        _ensure_required_folder_and_files_exists(default_dest, device_name)
        exported_config = GptJsonConfig.parse_file(
            default_dest / device_name / "config.json"
        )
        assert exported_config.model_config.max_batch_size == 1
        assert exported_config.model_config.max_beam_width == 1
        assert exported_config.model_config.max_input_len >= 1


@pytest.mark.parametrize(
    "recipe",
    [
        "qawq_recipe.py",
        "qfloat8_and_kv_cache_recipe.py",
    ],
)
def test_optimum_export_with_quantization(
    script_runner: "ScriptRunner", recipe: str
) -> None:
    cwd = Path(os.path.abspath(__file__)).parent
    recipe_path = cwd.parent / "fixtures" / recipe

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device_name = get_device_name(0)[-1]

    with TemporaryDirectory() as dest:
        default_dest = Path(dest)
        out = script_runner.run(
            f"optimum-cli export trtllm --quantization {recipe_path} --destination {default_dest} {model_id}",
            shell=True,
        )
        assert out.success

        _ensure_required_folder_and_files_exists(default_dest, device_name)
        exported_config = GptJsonConfig.parse_file(
            default_dest / device_name / "config.json"
        )
        assert exported_config.model_config.max_batch_size == 1
        assert exported_config.model_config.max_beam_width == 1
        assert exported_config.model_config.max_input_len >= 1
        assert exported_config.model_config.quant_mode != QuantMode.none()

        # Prune the cache
        default_dest = Workspace.from_hub_cache(model_id, device_name)
        shutil.rmtree(default_dest.root)
