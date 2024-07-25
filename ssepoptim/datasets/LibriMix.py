from typing import Literal

from ssepoptim.dataset import (
    SpeechSeparationDataset,
    SpeechSeparationDatasetConfig,
    SpeechSeparationDatasetFactory,
)
from ssepoptim.utils.type_checker import check_config_entries


class LibriMixDatasetConfig(SpeechSeparationDatasetConfig):
    pass


class LibriMixDataset(SpeechSeparationDataset):
    _DATASET_TYPE = SpeechSeparationDataset._DATASET_TYPE

    def __init__(self, config: LibriMixDatasetConfig):
        self._config = config

    def _get(self, subfolder: Literal["tr", "cv", "tt"]) -> _DATASET_TYPE:
        raise NotImplementedError()

    def get_train(self) -> _DATASET_TYPE:
        return self._get("tr")

    def get_valid(self) -> _DATASET_TYPE:
        return self._get("cv")

    def get_test(self) -> _DATASET_TYPE:
        return self._get("tt")


class LibriMixDatasetFactory(SpeechSeparationDatasetFactory):
    @staticmethod
    def _get_config():
        return LibriMixDatasetConfig

    @staticmethod
    def _get_object(config: SpeechSeparationDatasetConfig):
        return LibriMixDataset(check_config_entries(config, LibriMixDatasetConfig))
