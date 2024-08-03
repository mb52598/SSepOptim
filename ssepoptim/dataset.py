from abc import ABCMeta, abstractmethod
from typing import TypeVar

import torch
from torch.utils.data import Dataset

from ssepoptim.base.configuration import BaseConfig
from ssepoptim.base.factory import Factory

T = TypeVar("T")


class LenDataset(Dataset[T]):
    @abstractmethod
    def __len__(self) -> int: ...


class SpeechSeparationDatasetConfig(BaseConfig):
    path: str


class SpeechSeparationDataset(metaclass=ABCMeta):
    _DATASET_TYPE = LenDataset[tuple[torch.Tensor, torch.Tensor]]

    @abstractmethod
    def get_train(self) -> _DATASET_TYPE: ...

    @abstractmethod
    def get_valid(self) -> _DATASET_TYPE: ...

    @abstractmethod
    def get_test(self) -> _DATASET_TYPE: ...


class SpeechSeparationDatasetFactory(
    Factory[SpeechSeparationDatasetConfig, SpeechSeparationDataset], metaclass=ABCMeta
):
    @staticmethod
    @abstractmethod
    def _download(folder_path: str) -> None: ...

    @classmethod
    def download(cls, name: str, folder_path: str) -> None:
        return cls._get_subclass(name)._download(folder_path)
