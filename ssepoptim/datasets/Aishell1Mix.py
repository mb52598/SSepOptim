import os
import shutil
import subprocess
import sys
from typing import Literal

from ssepoptim.dataset import (
    SpeechSeparationDataset,
    SpeechSeparationDatasetConfig,
    SpeechSeparationDatasetFactory,
)
from ssepoptim.datasets.utils.csv_dataset import CsvAudioDataset
from ssepoptim.utils.type_checker import check_config_entries


class Aishell1MixDatasetConfig(SpeechSeparationDatasetConfig):
    path: str


class Aishell1MixDataset(SpeechSeparationDataset):
    _DATASET_TYPE = SpeechSeparationDataset._DATASET_TYPE

    def __init__(self, config: Aishell1MixDatasetConfig):
        self._config = config

    def _get(self, subfolder: Literal["tr", "cv", "tt"]) -> _DATASET_TYPE:
        metadata_path = os.path.join(
            self._config["path"],
            "aishell1mix",
            "Aishell1Mix2",
            "wav8k",
            "min",
            "metadata",
        )
        match subfolder:
            case "tr":
                csv_path = os.path.join(metadata_path, "mixture_train_mix_both.csv")
            case "cv":
                csv_path = os.path.join(metadata_path, "mixture_dev_mix_both.csv")
            case "tt":
                csv_path = os.path.join(metadata_path, "mixture_test_mix_both.csv")
        return CsvAudioDataset(
            csv_path,
            "mixture_path",
            ["source_1_path", "source_2_path"],
        )

    def get_train(self) -> _DATASET_TYPE:
        return self._get("tr")

    def get_valid(self) -> _DATASET_TYPE:
        return self._get("cv")

    def get_test(self) -> _DATASET_TYPE:
        return self._get("tt")


class Aishell1MixDatasetFactory(SpeechSeparationDatasetFactory):
    @staticmethod
    def _download(folder_path: str) -> None:
        # Create dirs
        os.makedirs(folder_path, exist_ok=True)
        # Setup paths
        repo_path = os.path.join(folder_path, "Aishell1Mix_git")
        dataset_path = os.path.join(folder_path, "Aishell1Mix")
        # Clone repo if it doesn't exist
        if not os.path.exists(repo_path):
            clone_retcode = subprocess.call(
                [
                    "git",
                    "clone",
                    "https://github.com/huangzj421/Aishell1Mix",
                    repo_path,
                ]
            )
            if clone_retcode != 0:
                raise RuntimeError(
                    f"Unable to clone Aishell1Mix git repository, errcode {clone_retcode}"
                )
        # Chmod download script
        script_path = os.path.join(repo_path, "generate_aishell1mix.sh")
        chmod_retcode = subprocess.call(["chmod", "+x", script_path])
        if chmod_retcode != 0:
            raise RuntimeError(
                f'Unable to chmod Aishell1Mix download script at path "{script_path}", errcode {chmod_retcode}'
            )
        # Change the script
        sed_retcode = 0
        # TODO: Remove url swap once the script is fixed
        old_url = (
            "https://storage.googleapis.com/whisper-public/wham_noise.zip".replace(
                "/", "\\/"
            )
        )
        new_url = "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip".replace(
            "/", "\\/"
        )
        sed_retcode += subprocess.call(
            [
                "sed",
                "-i",
                f"s/{old_url}/{new_url}/",
                script_path,
            ]
        )
        python_path = sys.executable.replace("/", "\\/")
        sed_retcode += subprocess.call(
            [
                "sed",
                "-i",
                f's/python /"{python_path}" /',
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
                f's/scripts\\/create_aishell1mix_from_metadata.py /"{scripts_path}\\/create_aishell1mix_from_metadata.py" /',
                script_path,
            ]
        )
        metadata_path = os.path.join(repo_path, "metadata", "aishell1mix").replace(
            "/", "\\/"
        )
        sed_retcode += subprocess.call(
            [
                "sed",
                "-i",
                f's/metadata_dir=$aishell1mix_md_dir/metadata_dir="{metadata_path}"/',
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
            ["sed", "-i", "s/--modes max min/--modes min/", script_path]
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
            raise RuntimeError("Failed to change Aishell1Mix script")
        # Call the download script
        script_retcode = subprocess.call([script_path, dataset_path])
        if script_retcode != 0:
            raise RuntimeError(
                f"Unable to execute Aishell1Mix script, errcode {script_retcode}"
            )
        # Remove repo
        shutil.rmtree(repo_path)

    @staticmethod
    def _get_config():
        return Aishell1MixDatasetConfig

    @staticmethod
    def _get_object(config: SpeechSeparationDatasetConfig):
        return Aishell1MixDataset(
            check_config_entries(config, Aishell1MixDatasetConfig)
        )
