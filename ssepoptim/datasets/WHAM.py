import os
from typing import Literal

import torch

from ssepoptim.dataset import (
    SpeechSeparationDataset,
    SpeechSeparationDatasetConfig,
    SpeechSeparationDatasetFactory,
)
from ssepoptim.datasets.utils import CsvAudioDataset
from ssepoptim.utils.checker import check_config_entries


class SpecialDataset(CsvAudioDataset):
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = super().__getitem__(idx)
        return item[0][0], item[1]


class WHAMDatasetConfig(SpeechSeparationDatasetConfig):
    pass


class WHAMDataset(SpeechSeparationDataset):
    _DATASET_TYPE = SpeechSeparationDataset._DATASET_TYPE

    def __init__(self, config: WHAMDatasetConfig):
        self._config = config

    def _get(self, subfolder: Literal["tr", "cv", "tt"]) -> _DATASET_TYPE:
        dataset_path = os.path.join(self._config["path"], "wham_noise")
        dataset = SpecialDataset(
            os.path.join(dataset_path, "metadata", f"mix_param_meta_{subfolder}.csv"),
            "utterance_id",
            os.path.join(dataset_path, subfolder),
        )
        return dataset

    def get_train(self) -> _DATASET_TYPE:
        return self._get("tr")

    def get_valid(self) -> _DATASET_TYPE:
        return self._get("cv")

    def get_test(self) -> _DATASET_TYPE:
        return self._get("tt")


class WHAMDatasetFactory(SpeechSeparationDatasetFactory):
    @staticmethod
    def _get_config():
        return WHAMDatasetConfig

    @staticmethod
    def _get_object(config: SpeechSeparationDatasetConfig):
        return WHAMDataset(check_config_entries(config, WHAMDatasetConfig))
