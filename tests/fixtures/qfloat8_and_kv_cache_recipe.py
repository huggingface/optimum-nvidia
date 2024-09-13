from typing import TYPE_CHECKING, Iterable

import numpy as np
import torch
from datasets import load_dataset
from modelopt.torch.quantization import FP8_DEFAULT_CFG, QuantizeConfig

from optimum.nvidia.compression.modelopt import ModelOptConfig, ModelOptRecipe


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class CiC4NewModelOptRecipe(ModelOptRecipe):
    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
    ):
        self._tokenizer = tokenizer
        self._nb_samples = 32

    @property
    def config(self) -> ModelOptConfig:
        config = FP8_DEFAULT_CFG.copy()
        config["quant_cfg"]["*output_quantizer"] = {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True,
        }

        qconfig = QuantizeConfig(**config)
        return ModelOptConfig(qconfig, sparsity=None)

    @property
    def dataset(self) -> Iterable:
        data_files = {"train": "en/c4-train.00000-of-01024.json.gz"}
        data = load_dataset("allenai/c4", split="train", data_files=data_files)
        indexes = np.random.choice(data.num_rows, size=self._nb_samples)
        encodings = self._tokenizer(
            data[indexes]["text"],
            truncation=True,
            max_length=2048,
            return_attention_mask=False,
        )

        dataset = [
            {
                "input_ids": torch.tensor(tokens.ids, dtype=torch.long, device="cuda")[
                    None
                ]
            }
            for tokens in encodings.encodings
        ]

        return dataset


TARGET_QUANTIZATION_RECIPE = CiC4NewModelOptRecipe
