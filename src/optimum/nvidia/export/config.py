from dataclasses import dataclass
from logging import getLogger
from typing import Optional, TYPE_CHECKING, NamedTuple, Union
from warnings import warn

from optimum.utils import NormalizedConfig
from tensorrt_llm import BuildConfig, Mapping
from tensorrt_llm.plugin import PluginConfig

from optimum.nvidia.lang import DataType
from optimum.nvidia.quantization import Float8QuantizationConfig

if TYPE_CHECKING:
    from datasets import Dataset
    from optimum.nvidia.quantization import AmmoQuantizationConfig
    from transformers import PretrainedConfig

INFER_NUM_LOCAL_GPUS = -1
LOGGER = getLogger()


@dataclass
class ShardingInfo:
    auto_parallel: bool
    tp: int
    pp: int
    world_size: int

    def to_mapping(self) -> Mapping:
        if self.auto_parallel:
            # auto-parallel is run at build time and requires a single shard export
            return Mapping(tp_size=1, pp_size=1, world_size=1)
        else:
            return Mapping(tp_size=self.tp, pp_size=self.pp, world_size=self.world_size)


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
    quantization: Optional["AmmoQuantizationConfig"] = None

    optimization_level: int = 3

    def __post_init__(self):
        if self.max_batch_size < 1:
            raise ValueError(f"max_batch_size should >= 1, got {self.max_batch_size}")

    @staticmethod
    def from_config(config: Union[NormalizedConfig, "PretrainedConfig"], max_batch_size: int = 32) -> "ExportConfig":
        if not isinstance(config, NormalizedConfig):
            config = NormalizedConfig(config)

        dtype = DataType.from_torch(config.torch_dtype).value
        max_input_len = config.max_position_embeddings
        max_output_len = config.max_position_embeddings

        return ExportConfig(
            dtype=dtype,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_batch_size=max_batch_size
        )

    def validate(self):
        if self.optimization_level < 0:
            raise ValueError(f"optimization_level should be >= 0, got {self.optimization_level}")

        if self.max_num_tokens == -1:
            if self.enabled_chunked_context:
                # Should be N * tokens_per_block
                self.max_num_tokens = 128  # hardcode for now
                warn(f"max_num_tokens set to {self.max_num_tokens} with chunked context enabled might not be optimal.")
            else:
                self.max_num_tokens = 2 * self.max_input_len

            LOGGER.debug(f"Inferred max_num_tokens={self.max_num_tokens}")

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

    def to_builder_config(self, plugin_config: Optional[PluginConfig] = None) -> "BuildConfig":
        self.validate()

        return BuildConfig(
            max_input_len=self.max_input_len,
            max_output_len=self.max_output_len,
            max_batch_size=self.max_batch_size,
            max_beam_width=self.max_beam_width,
            max_num_tokens=self.max_num_tokens,
            builder_opt=self.optimization_level,
            plugin_config=plugin_config or self.plugin_config
        )

    def with_sharding(self, tp: int = 1, pp: int = 1, sharding: Optional[ShardingInfo] = None) -> "ExportConfig":
        self.sharding = sharding or ShardingInfo(auto_parallel=False, tp=tp, pp=pp, world_size=tp * pp)
        return self

    def with_quantization(self, qconfig: "AmmoQuantizationConfig") -> "ExportConfig":
        self.quantization = qconfig
        return self


def auto_parallel(config: "ExportConfig", world_size: int = INFER_NUM_LOCAL_GPUS) -> "ExportConfig":
    """
    Helper to infer the most suitable parallelization strategy to apply to the model with respect to the local hardware.
    :param config: `ExportConfig` the quantization process should be added to
    :param world_size: Number of GPUs to consider when discovering automatic parallelization strategies
    :return: `ExportConfig`
    """

    if world_size < 1:
        from optimum.nvidia.utils.nvml import get_device_count
        world_size = get_device_count()

    LOGGER.info(f"Creating auto-parallelization strategy on {world_size}-GPUs")
    return config.with_sharding(sharding=ShardingInfo(auto_parallel=True, tp=1, pp=1, world_size=world_size))


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
        sharding=ShardingInfo(auto_parallel=False, tp=tp, pp=pp, world_size=tp * pp)
    )


def float8(
    config: "ExportConfig",
    kv_cache: bool = True,
    quantize_lm_head: bool = False,
    calibration_dataset: Optional["Dataset"] = None,
):
    """
    Define a float8 quantization step to apply to the model when building TRTLLM checkpoints
    :param config: `ExportConfig` the quantization process should be added to
    :param kv_cache: Flag indicating the KV cache will store values in float8 precision
    :param quantize_lm_head: Flag indicating if the final language model head should also be quantized to float8
    :param  calibration_dataset : Optional preprocessed dataset used to calibrate activations during quantization
    :return: `ExportConfig`
    """
    qconfig = Float8QuantizationConfig(
        with_quantized_kv_cache=kv_cache,
        with_quantized_lm_head=quantize_lm_head,
        calibration_data=calibration_dataset
    )
    return config.with_quantization(qconfig)
