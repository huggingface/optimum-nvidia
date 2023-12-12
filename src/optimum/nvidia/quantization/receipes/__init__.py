from datasets import Dataset

from optimum.nvidia.quantization import Calibration


def get_default_calibration_dataset(num_calibration_samples: int = 512) -> Calibration:
    from .cnn_daily import get_cnn_daily_calibration_dataset

    return get_cnn_daily_calibration_dataset(num_calibration_samples)
