from abc import ABC, abstractmethod
from logging import getLogger

from datasets import IterableDataset, load_dataset
from transformers import PreTrainedTokenizer


LOGGER = getLogger(__name__)


class Calibration(ABC):
    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

class HfDatasetCalibration(Calibration):
    @classmethod
    def from_datasets(cls, dataset: str, split: str, num_samples: int, column: str, streaming: bool = True, **kwargs):
        dataset = load_dataset(dataset, split=split, streaming=streaming, **kwargs)
        dataset = dataset.take(num_samples).select_columns([column])

        return cls(dataset)

    def __init__(self, dataset: IterableDataset):
        self._dataset = dataset

    def __iter__(self):
        return self._dataset.iter(batch_size=1)

    def tokenize(self, tokenizer: PreTrainedTokenizer, max_length: int = None, pad_to_multiple_of: int = 1):
        fieldname = self._dataset.column_names[0]

        if not tokenizer.pad_token_id and tokenizer.eos_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self._dataset = self._dataset.map(
            lambda x: tokenizer(
                x[fieldname],
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                truncation=True,
                padding=tokenizer.pad_token_id is not None,
                return_tensors="pt",
            )
        ).remove_columns(fieldname)

