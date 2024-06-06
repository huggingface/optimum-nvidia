from dataclasses import dataclass
from logging import getLogger
from typing import Optional, TYPE_CHECKING, NamedTuple, Union

from transformers import PretrainedConfig

from optimum.nvidia.quantization import AmmoQuantizationConfig, Float8QuantizationConfig

if TYPE_CHECKING:
    from datasets import Dataset

INFER_NUM_LOCAL_GPUS = -1
LOGGER = getLogger()


ShardingInfo = NamedTuple(
    "ShardingInfo", [
        ("auto_parallel", bool),
        ("tp", int),
        ("pp", int),
        ("world_size", int)
    ]
)


@dataclass
class ExportConfig:
    sharding: Optional[ShardingInfo] = None
    quantization: Optional["AmmoQuantizationConfig"] = None

    def with_sharding(self, tp: int = 1, pp: int = 1, sharding: Optional[ShardingInfo] = None) -> "ExportConfig":
        self.sharding = sharding or ShardingInfo(auto_parallel=False, tp=tp, pp=pp, world_size=tp * pp)
        return self

    def with_quantization(self, qconfig: "AmmoQuantizationConfig") -> "ExportConfig":
        self.quantization = qconfig
        return self


def default(model_config: PretrainedConfig) -> "ExportConfig":
    config = ExportConfig()
    return config


def auto_parallel(
    config: Optional["ExportConfig"] = None,
    world_size: int = INFER_NUM_LOCAL_GPUS,
) -> "ExportConfig":
    """
    Helper to infer the most suitable parallelization strategy to apply to the model with respect to the local hardware.
    :param config: Optional `ExportConfig` the quantization process should be added to
    :param world_size: Number of GPUs to consider when discovering automatic parallelization strategies
    :return: `ExportConfig`
    """

    if world_size < 1:
        from optimum.nvidia.utils.nvml import get_device_count
        world_size = get_device_count()

    return (config or ExportConfig()).with_sharding(
        sharding=ShardingInfo(auto_parallel=True, tp=1, pp=1, world_size=world_size)
    )


def sharded(
    config: Optional["ExportConfig"] = None,
    tp: int = 1,
    pp: int = 1
) -> "ExportConfig":
    """
    Helper to specific the parallelization strategy to apply to the model
    :param config: Optional `ExportConfig` the quantization process should be added to
    :param tp: Tensor Parallelism degree to apply (`int` >= 1)
    :param pp: Pipeline Parallelism degree to apply (`int` >= 1)
    :return: `ExportConfig`
    """
    if tp < 1:
        raise ValueError(f"Tensor Parallelism (tp) should be >= 1 (got: tp={tp})")

    if pp < 1:
        raise ValueError(f"Pipeline Parallelism (pp) should be >= 1 (got: pp={pp})")

    return (config or ExportConfig()).with_sharding(
        sharding=ShardingInfo(auto_parallel=False, tp=tp, pp=pp, world_size=tp * pp)
    )


def float8(
    config: Optional["ExportConfig"] = None,
    kv_cache: bool = True,
    quantize_lm_head: bool = False,
    calibration_dataset: Optional["Dataset"] = None,
):
    """
    Define a float8 quantization step to apply to the model when building TRTLLM checkpoints
    :param config: Optional `ExportConfig` the quantization process should be added to
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
    return (config or ExportConfig()).with_quantization(qconfig)
