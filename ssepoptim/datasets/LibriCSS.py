from typing import Literal

from ssepoptim.dataset import (
    SpeechSeparationDataset,
    SpeechSeparationDatasetConfig,
    SpeechSeparationDatasetFactory,
)
from ssepoptim.utils.checker import check_config_entries


class LibriCSSDatasetConfig(SpeechSeparationDatasetConfig):
    pass


class LibriCSSDataset(SpeechSeparationDataset):
    _DATASET_TYPE = SpeechSeparationDataset._DATASET_TYPE

    def __init__(self, config: LibriCSSDatasetConfig):
        self._config = config

    def _get(self, subfolder: Literal["tr", "cv", "tt"]) -> _DATASET_TYPE:
        raise NotImplementedError()

    def get_train(self) -> _DATASET_TYPE:
        return self._get("tr")

    def get_valid(self) -> _DATASET_TYPE:
        return self._get("cv")

    def get_test(self) -> _DATASET_TYPE:
        return self._get("tt")


class LibriCSSDatasetFactory(SpeechSeparationDatasetFactory):
    @staticmethod
    def _get_config():
        return LibriCSSDatasetConfig

    @staticmethod
    def _get_object(config: SpeechSeparationDatasetConfig):
        return LibriCSSDataset(check_config_entries(config, LibriCSSDatasetConfig))
