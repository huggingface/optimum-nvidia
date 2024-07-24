from dataclasses import dataclass
from logging import getLogger
from os import PathLike
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn

from tensorrt_llm import BuildConfig
from tensorrt_llm import Mapping as ShardingInfo
from tensorrt_llm.plugin import PluginConfig
from transformers import AutoConfig

from optimum.nvidia.lang import DataType
from optimum.utils import NormalizedConfig


if TYPE_CHECKING:
    from transformers import PretrainedConfig

INFER_NUM_LOCAL_GPUS = -1
LOGGER = getLogger()


@dataclass
class ExportConfig:
    dtype: str
    max_input_len: int
    max_output_len: int
    max_batch_size: int

    # Optional parameters
    max_beam_width: int = 1
    max_num_tokens: int = -1
    enabled_chunked_context: int = False

    sharding: Optional[ShardingInfo] = None

    optimization_level: int = 3

    def __post_init__(self):
        if self.max_batch_size < 1:
            raise ValueError(f"max_batch_size should >= 1, got {self.max_batch_size}")

    @staticmethod
    def from_pretrained(
        model_id_or_path: Union[str, PathLike], max_batch_size: int = 1
    ) -> "ExportConfig":
        return ExportConfig.from_config(
            AutoConfig.from_pretrained(model_id_or_path), max_batch_size
        )

    @staticmethod
    def from_config(
        config: Union[NormalizedConfig, "PretrainedConfig"], max_batch_size: int = 1
    ) -> "ExportConfig":
        if not isinstance(config, NormalizedConfig):
            config = NormalizedConfig(config)

        dtype = DataType.from_torch(config.torch_dtype).value
        max_input_len = config.max_position_embeddings
        max_output_len = config.max_position_embeddings

        econfig = ExportConfig(
            dtype=dtype,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_batch_size=max_batch_size,
        )

        # Initialize sharing information with single shard
        econfig.with_sharding()
        econfig.validate()
        return econfig

    def validate(self) -> "ExportConfig":
        if self.optimization_level < 0:
            raise ValueError(
                f"optimization_level should be >= 0, got {self.optimization_level}"
            )

        if self.max_num_tokens == -1:
            if self.enabled_chunked_context:
                # Should be N * tokens_per_block
                self.max_num_tokens = 128  # hardcode for now
                warn(
                    f"max_num_tokens set to {self.max_num_tokens} with chunked context enabled might not be optimal."
                )
            else:
                self.max_num_tokens = 2 * self.max_input_len

            LOGGER.debug(f"Inferred max_num_tokens={self.max_num_tokens}")

        return self

    @property
    def plugin_config(self) -> "PluginConfig":
        config = PluginConfig()
        config.gemm_plugin = self.dtype
        config.bert_attention_plugin = self.dtype
        config.gpt_attention_plugin = self.dtype
        config.nccl_plugin = self.dtype
        config.mamba_conv1d_plugin = self.dtype
        config.moe_plugin = self.dtype
        return config

    def to_builder_config(
        self, plugin_config: Optional[PluginConfig] = None
    ) -> "BuildConfig":
        self.validate()

        return BuildConfig(
            max_input_len=self.max_input_len,
            max_seq_len=self.max_output_len,
            max_batch_size=self.max_batch_size,
            max_beam_width=self.max_beam_width,
            max_num_tokens=self.max_num_tokens,
            builder_opt=self.optimization_level,
            plugin_config=plugin_config or self.plugin_config,
        )

    def with_sharding(
        self,
        tp: int = 1,
        pp: int = 1,
        gpus_per_node: int = 8,
        sharding: Optional[ShardingInfo] = None,
    ) -> "ExportConfig":
        self.sharding = sharding or ShardingInfo(
            tp_size=tp, pp_size=pp, world_size=tp * pp, gpus_per_node=gpus_per_node
        )
        return self


def auto_parallel(
    config: "ExportConfig", world_size: int = INFER_NUM_LOCAL_GPUS
) -> "ExportConfig":
    """
    Helper to infer the most suitable parallelization strategy to apply to the model with respect to the local hardware.
    :param config: `ExportConfig` the quantization process should be added to
    :param world_size: Number of GPUs to consider when discovering automatic parallelization strategies
    :return: `ExportConfig`
    """
    # Infer number of GPUs on the system
    if world_size < 1:
        from optimum.nvidia.utils.nvml import get_device_count

        world_size = get_device_count()

        LOGGER.info(f"Found {world_size} GPUs on the system")

    # Handle all the different cases (0, 1, N > 1)
    if world_size == 0:
        raise ValueError("No GPU found")
    elif world_size == 1:
        return config.with_sharding(tp=1, pp=1, gpus_per_node=world_size)
    else:
        LOGGER.info(f"Creating auto-parallelization strategy on {world_size}-GPUs")
        LOGGER.warning(
            "Auto-parallelization strategy is currently in beta and might not be optimal"
        )

        if world_size == 2:
            return config.with_sharding(tp=2, pp=1, gpus_per_node=world_size)
        elif world_size == 4:
            return config.with_sharding(tp=2, pp=2, gpus_per_node=world_size)
        elif world_size == 8:
            return config.with_sharding(tp=4, pp=2, gpus_per_node=world_size)
        else:
            raise ValueError(
                f"Unsupported number of GPUs: {world_size}. "
                "Please open-up and issue on the optimum-nvidia repository: "
                "https://github.com/huggingface/optimum-nvidia"
            )


def sharded(config: "ExportConfig", tp: int = 1, pp: int = 1) -> "ExportConfig":
    """
    Helper to specific the parallelization strategy to apply to the model
    :param config: `ExportConfig` the quantization process should be added to
    :param tp: Tensor Parallelism degree to apply (`int` >= 1)
    :param pp: Pipeline Parallelism degree to apply (`int` >= 1)
    :return: `ExportConfig`
    """
    if tp < 1:
        raise ValueError(f"Tensor Parallelism (tp) should be >= 1 (got: tp={tp})")

    if pp < 1:
        raise ValueError(f"Pipeline Parallelism (pp) should be >= 1 (got: pp={pp})")

    return config.with_sharding(
        sharding=ShardingInfo(tp_size=tp, pp_size=pp, world_size=tp * pp)
    )
