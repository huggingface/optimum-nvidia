from itertools import chain
from logging import getLogger
from pathlib import Path
from subprocess import run
from typing import Any, Dict

from tensorrt_llm.models import PretrainedConfig

from optimum.nvidia.builder.config import EngineConfig


LOGGER = getLogger()


class LocalEngineBuilder:

    TRTLLM_BUILD_EXEC = "trtllm-build"

    @staticmethod
    def build_cli_command(root: Path, config: EngineConfig) -> Dict[str, Any]:
        workload_params = {
            "--max_batch_size": config.workload_profile.max_batch_size,
            "--max_input_len": config.workload_profile.max_input_len,
            "--max_output_len": config.workload_profile.max_output_len,
        }

        generation_params = {
            "--max_beam_width": config.generation_profile.num_beams,
            "--max_num_tokens": config.generation_profile.max_new_tokens
        }

        if config.generation_profile.max_draft_length >= 1:
            generation_params["--max_draft_len"] = config.generation_profile.max_draft_length

        build_params = {
            "--checkpoint_dir": root,
            "--output_dir": root,
            "--model_config": root / "config.json",
            "--builder_opt": config.optimisation_level,
            "--strongly_typed": None,
            "--logits_dtype": config.logits_dtype
        }

        return build_params | generation_params | workload_params

    def __init__(self, config: PretrainedConfig, output_folder: Path):
        self._config = config
        self._output_folder = output_folder

    def build(self, config: EngineConfig):
        cli_params = LocalEngineBuilder.build_cli_command(self._output_folder, config)
        cli_params_list = [str(t) for t in chain.from_iterable(cli_params.items())]
        cli_params_list = [i for i in cli_params_list if i != "None"]

        LOGGER.debug(f"trtllm-build parameters: {cli_params_list}")

        for rank in range(self._config.mapping.world_size):
            ranked_checkpoint = f"rank{rank}.safetensors"
            if not (self._output_folder / ranked_checkpoint).exists():
                raise ValueError(f"Missing rank-{rank} checkpoints (rank{rank}.safetensors), cannot build.")

        run([LocalEngineBuilder.TRTLLM_BUILD_EXEC] + cli_params_list)
