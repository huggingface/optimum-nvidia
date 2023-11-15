from abc import ABC, abstractmethod
from logging import getLogger

from datasets import load_dataset, IterableDataset
from transformers import PreTrainedTokenizer

LOGGER = getLogger(__name__)


class Calibration(ABC):

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()


# class HfDatasetCalibration(Calibration):
#
#     def __init__(self, dataset: str, split: str, num_samples: int, preprocessing_functor: Callable, streaming: bool = True):
#         self._num_samples = num_samples
#         self._functor = preprocessing_functor
#         self._dataset = load_dataset(dataset, split=split, streaming=streaming) \
#             .take(num_samples) \
#             .map(preprocessing_functor) \
#
#     def __len__(self):
#         return self._num_samples
#
#     def __iter__(self):
#         return self._dataset.iter(batch_size=1)


class HfDatasetCalibration(Calibration):

    @classmethod
    def from_datasets(cls, name: str, split: str, num_samples: int, column: str, streaming: bool = True):
        dataset = load_dataset(name, split=split, streaming=streaming)
        dataset = dataset.take(num_samples).select_columns([column])

        return cls(dataset)

    def __init__(self, dataset: IterableDataset):
        self._dataset = dataset

    def __iter__(self):
        return self._dataset.iter(batch_size=1)

    def tokenize(self, tokenizer: PreTrainedTokenizer, max_length: int = None, pad_to_multiple_of: int = 1):
        fieldname = self._dataset.column_names[0]
        self._dataset = self._dataset.map(
            lambda x: tokenizer(
                x[fieldname],
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt"
            )
        ).remove_columns(fieldname)


# @runtime_checkable
# class SupportsWeightQuantization(Protocol):
#
#     QUANTIZATION_EXCLUDED_PARAMETERS: Optional[Set[str]]
#
#     @staticmethod
#     @property
#     def named_weight_parameters() -> Iterable[Tuple[str, np.array]]:
#         ...
#
#
#
# class SupportsWeightAndActivationQuantization(Protocol):
#     pass
#
#
# QUANTIZATION_PROTOCOLS = {SupportsWeightQuantization}
