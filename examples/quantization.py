#  coding=utf-8
#  Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from modelopt.torch.quantization import (
    FP8_WA_FP8_KV_CFG,
    INT4_AWQ_REAL_QUANT_CFG,
    W4A8_AWQ_BETA_CFG,
    QuantizeConfig,
)
from modelopt.torch.sparsity.config import SparseGPTConfig, SparseMagnitudeConfig
from tensorrt_llm.quantization.quantize_by_modelopt import KV_CACHE_CFG
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, GenerationConfig

from optimum.nvidia import AutoModelForCausalLM, ExportConfig, setup_logging
from optimum.nvidia.compression.modelopt import (
    ModelOptConfig,
    ModelOptQuantizer,
    ModelOptRecipe,
)


# Setup logging needs to happen before importing TRT ...
setup_logging(True)

LOGGER = getLogger(__name__)


class ExampleC4NewModelOptRecipe(ModelOptRecipe):
    @staticmethod
    def awq(
        tokenizer: PreTrainedTokenizer,
        num_samples: int = 512
    ) -> "ExampleC4NewModelOptRecipe":
        return ExampleC4NewModelOptRecipe(
            ModelOptConfig(QuantizeConfig(**INT4_AWQ_REAL_QUANT_CFG)),
            tokenizer,
            num_samples,
        )

    @staticmethod
    def float8(
        tokenizer: PreTrainedTokenizer,
        sparsity: Optional[Union[SparseGPTConfig, SparseMagnitudeConfig]] = None,
        num_samples: int = 512,
        use_float8_kv_cache: bool = True
    ) -> "ExampleC4NewModelOptRecipe":
        config = FP8_WA_FP8_KV_CFG
        if use_float8_kv_cache:
            fp8_kv_config = KV_CACHE_CFG.copy()
            for value in fp8_kv_config.values():
                value.update({"num_bits": (4, 3)})
            config["quant_cfg"].update(fp8_kv_config)

        return ExampleC4NewModelOptRecipe(
            ModelOptConfig(QuantizeConfig(**config), sparsity), tokenizer, num_samples
        )

    @staticmethod
    def w4a8(
        tokenizer: PreTrainedTokenizer,
        num_samples: int = 512
    ) -> "ExampleC4NewModelOptRecipe":
        return ExampleC4NewModelOptRecipe(
            ModelOptConfig(QuantizeConfig(**W4A8_AWQ_BETA_CFG)), tokenizer, num_samples
        )

    def __init__(
        self,
        config: ModelOptConfig,
        tokenizer: PreTrainedTokenizer,
        num_samples: int = 512,
    ):
        self._config = config
        self._tokenizer = tokenizer
        self._nb_samples = num_samples

    @property
    def config(self) -> ModelOptConfig:
        return self._config

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

        # return tqdm(dataset, desc="Quantizing...")
        return dataset


if __name__ == "__main__":
    parser = ArgumentParser("🤗 Optimum-Nvidia Custom Quantization Example")
    parser.add_argument(
        "--hub-token",
        type=str,
        help="Hugging Face Hub Token to retrieve private weights.",
    )

    parser.add_argument("--method", type=str, choices=["awq", "float8", "w4a8"])
    parser.add_argument(
        "--sparsity",
        type=str,
        choices=["sparsegpt", "sparse_magnitude"],
        help="Apply 2-4 SparseGPT sparsification"
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=512,
        help="Number of samples used for calibration",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help="Maximum concurrent request for the model",
    )
    parser.add_argument("model", type=str, help="The model's id or path to use")
    parser.add_argument(
        "output", type=Path, help="Path to store generated TensorRT engine"
    )
    args = parser.parse_args()

    if args.hub_token is not None:
        from huggingface_hub import login

        login(args.hub_token)

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization Config
    if args.method == "awq":
        qconfig = ExampleC4NewModelOptRecipe.awq(
            tokenizer, args.num_calibration_samples
        )
    elif args.method == "float8":
        qconfig = ExampleC4NewModelOptRecipe.float8(
            tokenizer, args.sparsity, args.num_calibration_samples
        )
    elif args.method == "w4a8":
        qconfig = ExampleC4NewModelOptRecipe.w4a8(
            tokenizer, args.num_calibration_samples
        )
    else:
        raise ValueError(
            f"Invalid quantization method {args.method}. Supported methods are (awq, float8, w4a8)"
        )

    quantizer = ModelOptQuantizer(qconfig)
    export = ExportConfig.from_pretrained(args.model, args.max_batch_size)

    # Create the model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        export_config=export,
        quantization_config=qconfig,
    )

    if not Path(args.output).exists():
        model.save_pretrained(args.output)

    generation_config = GenerationConfig.from_pretrained(args.model)
    generation_config.max_new_tokens = 256

    prompt = "What is the latest generation of Nvidia GPUs?"
    tokens = tokenizer(prompt, padding=True, return_tensors="pt")
    generated = model.generate(
        tokens["input_ids"],
        generation_config
    )

    generated_text = tokenizer.decode(
        generated, skip_special_tokens=True
    )
    print(generated_text)
