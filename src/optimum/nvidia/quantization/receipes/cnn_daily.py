from optimum.nvidia.quantization import Calibration, HfDatasetCalibration


def get_cnn_daily_calibration_dataset(num_calibration_samples: int = 512) -> Calibration:
    return HfDatasetCalibration.from_datasets(
        dataset="cnn_dailymail",
        name="3.0.0",
        split="train",
        num_samples=num_calibration_samples,
        column="article",
        streaming=True,
    )
