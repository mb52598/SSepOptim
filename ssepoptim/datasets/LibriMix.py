import os
import shutil
import subprocess
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
    def _download(folder_path: str) -> None:
        repo_path = os.path.join(folder_path, "LibriMix_git")
        dataset_path = os.path.join(folder_path, "LibriMix")
        clone_retcode = subprocess.call(
            [
                "git",
                "clone",
                "https://github.com/JorisCos/LibriMix",
                repo_path,
            ]
        )
        if clone_retcode != 0:
            raise RuntimeError(
                f"Unable to clone LibriMix git repository, errcode {clone_retcode}"
            )
        script_path = os.path.join(repo_path, "generate_librimix.sh")
        chmod_retcode = subprocess.call(["chmod", "+x", script_path])
        if chmod_retcode != 0:
            raise RuntimeError(
                f'Unable to chmod LibriMix download script at path "{script_path}", errcode {chmod_retcode}'
            )
        script_retcode = subprocess.call([script_path, dataset_path])
        if script_retcode != 0:
            raise RuntimeError(
                f"Unable to create LibriMix dataset, errcode {script_retcode}"
            )
        shutil.rmtree(repo_path)

    @staticmethod
    def _get_config():
        return LibriMixDatasetConfig

    @staticmethod
    def _get_object(config: SpeechSeparationDatasetConfig):
        return LibriMixDataset(check_config_entries(config, LibriMixDatasetConfig))
