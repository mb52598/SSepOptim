import os
import shutil
import subprocess
import sys
from typing import Literal, cast

import torch
from torch.utils.data import ConcatDataset

from ssepoptim.dataset import (
    LenDataset,
    SpeechSeparationDataset,
    SpeechSeparationDatasetConfig,
    SpeechSeparationDatasetFactory,
)
from ssepoptim.datasets.utils.csv_dataset import CsvAudioDataset
from ssepoptim.utils.type_checker import check_config_entries


class LibriMixDatasetConfig(SpeechSeparationDatasetConfig):
    path: str


class LibriMixDataset(SpeechSeparationDataset):
    _DATASET_TYPE = SpeechSeparationDataset._DATASET_TYPE

    def __init__(self, config: LibriMixDatasetConfig):
        self._config = config

    def _get(self, subfolder: Literal["tr", "cv", "tt"]) -> _DATASET_TYPE:
        metadata_path = os.path.join(
            self._config["path"], "Libri2Mix", "wav8k", "min", "metadata"
        )
        match subfolder:
            case "tr":
                csv_paths = [
                    os.path.join(metadata_path, "mixture_train-100_mix_both.csv"),
                    os.path.join(metadata_path, "mixture_train-360_mix_both.csv"),
                ]
            case "cv":
                csv_paths = [os.path.join(metadata_path, "mixture_dev_mix_both.csv")]
            case "tt":
                csv_paths = [os.path.join(metadata_path, "mixture_test_mix_both.csv")]
        return cast(
            LenDataset[tuple[torch.Tensor, torch.Tensor]],
            ConcatDataset(
                CsvAudioDataset(
                    csv_path,
                    "mixture_path",
                    ["source_1_path", "source_2_path"],
                )
                for csv_path in csv_paths
            ),
        )

    def get_train(self) -> _DATASET_TYPE:
        return self._get("tr")

    def get_valid(self) -> _DATASET_TYPE:
        return self._get("cv")

    def get_test(self) -> _DATASET_TYPE:
        return self._get("tt")


class LibriMixDatasetFactory(SpeechSeparationDatasetFactory):
    @staticmethod
    def _download(folder_path: str) -> None:
        # Create dirs
        os.makedirs(folder_path, exist_ok=True)
        # Setup paths
        repo_path = os.path.join(folder_path, "LibriMix_git")
        dataset_path = os.path.join(folder_path, "LibriMix")
        # Clone repo if it doesn't exist
        if not os.path.exists(repo_path):
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
        # Chmod download script
        script_path = os.path.join(repo_path, "generate_librimix.sh")
        chmod_retcode = subprocess.call(["chmod", "+x", script_path])
        if chmod_retcode != 0:
            raise RuntimeError(
                f'Unable to chmod LibriMix download script at path "{script_path}", errcode {chmod_retcode}'
            )
        # Change the script
        sed_retcode = 0
        python_path = sys.executable.replace("/", "\\/")
        sed_retcode += subprocess.call(
            [
                "sed",
                "-i",
                f's/python_path=python/python_path="{python_path}"/',
                script_path,
            ]
        )
        sed_retcode += subprocess.call(
            [
                "sed",
                "-i",
                's/$python_path /"$python_path" /',
                script_path,
            ]
        )
        scripts_path = os.path.join(repo_path, "scripts").replace("/", "\\/")
        sed_retcode += subprocess.call(
            [
                "sed",
                "-i",
                f's/scripts\\/augment_train_noise.py /"{scripts_path}\\/augment_train_noise.py" /',
                script_path,
            ]
        )
        sed_retcode += subprocess.call(
            [
                "sed",
                "-i",
                f's/scripts\\/create_librimix_from_metadata.py /"{scripts_path}\\/create_librimix_from_metadata.py" /',
                script_path,
            ]
        )
        metadata_path = os.path.join(repo_path, "metadata").replace("/", "\\/")
        sed_retcode += subprocess.call(
            [
                "sed",
                "-i",
                f's/metadata_dir=metadata/metadata_dir="{metadata_path}"/',
                script_path,
            ]
        )
        sed_retcode += subprocess.call(
            ["sed", "-i", "s/for n_src in 2 3; do/for n_src in 2; do/", script_path]
        )
        sed_retcode += subprocess.call(
            ["sed", "-i", "s/--freqs 8k 16k/--freqs 8k/", script_path]
        )
        sed_retcode += subprocess.call(
            ["sed", "-i", "s/--modes min max/--modes min/", script_path]
        )
        sed_retcode += subprocess.call(
            [
                "sed",
                "-i",
                "s/--types mix_clean mix_both mix_single/--types mix_both/",
                script_path,
            ]
        )
        if sed_retcode != 0:
            raise RuntimeError("Failed to change LibriMix script")
        # Call the download script
        script_retcode = subprocess.call([script_path, dataset_path])
        if script_retcode != 0:
            raise RuntimeError(
                f"Unable to execute LibriMix script, errcode {script_retcode}"
            )
        # Remove repo
        shutil.rmtree(repo_path)

    @staticmethod
    def _get_config():
        return LibriMixDatasetConfig

    @staticmethod
    def _get_object(config: SpeechSeparationDatasetConfig):
        return LibriMixDataset(check_config_entries(config, LibriMixDatasetConfig))
