from dataclasses import dataclass

from tensorrt_llm.quantization import QuantMode


NO_QUANTIZATION = QuantMode(0)


@dataclass
class QuantizationConfig:
    mode: QuantMode
    group_size: int = -1

    @property
    def has_quantization_step(self):
        return self.mode != NO_QUANTIZATION
