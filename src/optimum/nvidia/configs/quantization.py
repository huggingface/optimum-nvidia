from typing import Any, Dict, Set

from dataclasses import dataclass

from tensorrt_llm.quantization import QuantMode
from transformers import AwqConfig, GPTQConfig

from optimum.nvidia.configs import ModelConfig

NO_QUANTIZATION = QuantMode(0)

SUPPORTED_QUANTIZATION_METHODS = {"awq": AwqConfig, "gptq": GPTQConfig}


@dataclass
class QuantizationConfig:
    mode: QuantMode

    @staticmethod
    def from_model_config(config: ModelConfig):
        if "quantization_config" in config:
            qconfig = config["quantization_config"]
            qmethod = qconfig["quant_method"]

            if qmethod in SUPPORTED_QUANTIZATION_METHODS:

                # Retrieve the transformers quantization config
                qconfig = SUPPORTED_QUANTIZATION_METHODS[qmethod].from_dict(qconfig)

                if qmethod == "awq":
                    return AwqQuantizationConfig(qconfig)
                elif qmethod == "gptq":
                    raise ValueError("GPTQ is not yet supported")
            else:
                raise ValueError(f"Unsupported quantization method: {qmethod}")

        return QuantizationConfig.no_quantization()

    @staticmethod
    def no_quantization() -> "QuantizationConfig":
        return QuantizationConfig(NO_QUANTIZATION)

    @property
    def has_quantization_step(self) -> bool:
        return self.mode != NO_QUANTIZATION

    @property
    def has_calibration_step(self) -> bool:
        return self.mode.has_act_and_weight_quant()

    def to_quantizer_args(self) -> Dict[str, Any]:
        return dict()


class WeightOnlyQuantizationConfig(QuantizationConfig):
    num_bits: int
    group_size: int = -1

    @property
    def is_groupwise(self) -> bool:
        return self.group_size > 0

    def to_quantizer_args(self, exclude_modules: Set[str] = None) -> Dict[str, Any]:
        excluded_modules = list({"lm_head"} | (exclude_modules or set()))
        kwargs = {
            "exclude_modules": excluded_modules,
        }

        if self.group_size > 0:
            kwargs["group_size"] = self.group_size

        return kwargs


class AwqQuantizationConfig(WeightOnlyQuantizationConfig):
    def __init__(self, qconfig: AwqConfig):
        if qconfig.bits  in {4, 8}:
            self.num_bits = qconfig.bits
        else:
            raise ValueError(
                f"AWQ {qconfig.bits}-bits quantization is not supported at the moment. "
                "Please open-up an issue at https://github.com/huggingface/optimum-nvidia with your use-case"
            )

        if qconfig.group_size and qconfig.group_size > 0:
            self.group_size = qconfig.group_size
        else:
            self.group_size = -1

        self.mode = QuantMode.from_description(
            quantize_weights=True,
            quantize_activations=False,
            per_token=False,
            per_channel=False,
            per_group=self.group_size > 0,
            use_int4_weights=self.num_bits == 4,
            use_fp8_qdq=False,
            use_int8_kv_cache=False,
            use_fp8_kv_cache=False
        )

    def to_quantizer_args(self, exclude_modules: Set[str] = None) -> Dict[str, Any]:
        return super().to_quantizer_args() | {"zero": True, "pre_quant_scale": False}