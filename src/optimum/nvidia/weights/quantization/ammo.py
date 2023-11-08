from pathlib import Path
from typing import Type, Union, Optional, Dict

from datasets import Dataset
from huggingface_hub import ModelHubMixin
from huggingface_hub.hub_mixin import T
from torch import Module as TorchModule
from torch.utils.data import DataLoader
from tokenizers import Tokenizer


class AmmoQuantizer(ModelHubMixin):

    def from_pretrained(
        cls: Type[T],
        pretrained_model_name_or_path: Union[str, Path],
        *,
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Optional[Dict] = None,
        token: Optional[Union[str, bool]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        **model_kwargs,
    ) -> T:
        tokenizer = Tokenizer.from_pretrained(pretrained_model_name_or_path, revision)
        model = super().from_pretrained(
            cls,
            pretrained_model_name_or_path,
            force_download,
            resume_download,
            proxies,
            token,
            cache_dir,
            local_files_only,
            revision,
            **model_kwargs
        )

        return cls(tokenizer, model)


    def __init__(self, tokenizer: Tokenizer, model: TorchModule):
        self._tokenizer = tokenizer,
        self._model = model

    def calibrate(self, dataset: Dataset):
        for sample in DataLoader(dataset, batch_size=1):
            pass