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


SpeechSeparationDatasetType = LenDataset[tuple[torch.Tensor, torch.Tensor]]


class SpeechSeparationDataset(metaclass=ABCMeta):
    @abstractmethod
    def get_train(self) -> SpeechSeparationDatasetType: ...

    @abstractmethod
    def get_valid(self) -> SpeechSeparationDatasetType: ...

    @abstractmethod
    def get_test(self) -> SpeechSeparationDatasetType: ...


class SpeechSeparationDatasetFactory(
    Factory[SpeechSeparationDatasetConfig, SpeechSeparationDataset], metaclass=ABCMeta
):
    @staticmethod
    @abstractmethod
    def _download(folder_path: str) -> None: ...

    @classmethod
    def download(cls, name: str, folder_path: str) -> None:
        return cls._get_subclass(name)._download(folder_path)
