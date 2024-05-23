#  coding=utf-8
#  Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from itertools import chain
from logging import getLogger
from pathlib import Path
from subprocess import PIPE, STDOUT, run
from typing import Any, Dict

from optimum.nvidia import TensorRTConfig
from optimum.nvidia.builder.config import EngineConfig
from optimum.nvidia.utils.patching import BuilderPatcher


LOGGER = getLogger()
CLI_PLUGIN_NAMES = {
    # Plugins
    "bert_attention_plugin",
    "gpt_attention_plugin",
    "gemm_plugin",
    "lookup_plugin",
    "lora_plugin",
    "moe_plugin",
    # Features
    "context_fmha",
    "context_fmha_fp32_acc",
    "paged_kv_cache",
    "remove_input_padding",
    "use_custom_all_reduce",
    "multi_block_mode",
    "enable_xqa",
    "attention_qk_half_accumulation",
    "tokens_per_block",
    "use_paged_context_fmha",
    "use_context_fmha_for_generation",
}


def process_plugin_flag(_: str, value: Any) -> str:
    if isinstance(value, bool):
        return "enable" if value else "disable"
    else:
        return value


class LocalEngineBuilder:
    TRTLLM_BUILD_EXEC = "trtllm-build"

    @staticmethod
    def build_cli_command(
        checkpoints: Path,
        engines: Path,
        model_config: TensorRTConfig,
        build_config: EngineConfig,
    ) -> Dict[str, Any]:
        workload_params = {
            "--max_batch_size": build_config.workload_profile.max_batch_size,
            "--max_input_len": build_config.workload_profile.max_input_len,
            "--max_output_len": build_config.workload_profile.max_output_len,
        }

        generation_params = {
            "--max_beam_width": build_config.generation_profile.num_beams,
        }

        if build_config.generation_profile.max_draft_length >= 1:
            generation_params["--max_draft_len"] = (
                build_config.generation_profile.max_draft_length
            )

        plugins_params = {
            f"--{name}": process_plugin_flag(name, value)
            for name in CLI_PLUGIN_NAMES
            if (value := getattr(build_config.plugins_config, name)) is not None
        }

        build_params = {
            "--checkpoint_dir": checkpoints,
            "--output_dir": engines,
            "--model_config": checkpoints / "model.json",
            "--builder_opt": build_config.optimisation_level,
            "--logits_dtype": build_config.logits_dtype,
            "--tp_size": model_config.mapping.tp_size,
            "--pp_size": model_config.mapping.pp_size,
        }

        if hasattr(model_config, "trt_model_class") and hasattr(
            model_config, "trt_model_file"
        ):
            build_params["--model_cls_file"] = model_config.trt_model_file
            build_params["--model_cls_name"] = model_config.trt_model_class

        if model_config.supports_strong_typing():
            build_params["--strongly_typed"] = None

        return build_params | generation_params | workload_params | plugins_params

    def __init__(
        self, config: TensorRTConfig, checkpoint_folder: Path, output_folder: Path
    ):
        self._config = config
        self._checkpoint_folder = checkpoint_folder
        self._output_folder = output_folder

    def build(self, config: EngineConfig):
        cli_params = LocalEngineBuilder.build_cli_command(
            self._checkpoint_folder, self._output_folder, self._config, config
        )
        cli_params_list = [str(t) for t in chain.from_iterable(cli_params.items())]
        cli_params_list = [i for i in cli_params_list if i != "None"]

        LOGGER.info(f"trtllm-build parameters: {cli_params_list}")

        for rank in range(self._config.mapping.world_size):
            ranked_checkpoint = f"rank{rank}.safetensors"
            if not (self._checkpoint_folder / ranked_checkpoint).exists():
                raise ValueError(
                    f"Missing rank-{rank} checkpoints (rank{rank}.safetensors), cannot build."
                )

        # TODO: Remove BuilderPatcher once TensorRT-LLM updates its codebase to allow to disable `optimize(network)`.
        with BuilderPatcher():
            # Run the build
            result = run(
                [LocalEngineBuilder.TRTLLM_BUILD_EXEC] + cli_params_list,
                stdout=PIPE,
                stderr=STDOUT,
            )

        if result.returncode != 0:
            LOGGER.warning(
                f"trtllm-build stdout: {result.stdout.decode('utf-8') if result.stdout is not None else None}"
            )
            LOGGER.warning(
                f"trtllm-build stderr: {result.stderr.decode('utf-8') if result.stderr is not None else None}"
            )

            raise ValueError(
                f"Compilation failed ({result.returncode}), "
                "please open up an issue at https://github.com/huggingface/optimum-nvidia"
            )  # TODO: change with proper error
