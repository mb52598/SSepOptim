import os
from typing import Literal
from urllib.parse import urlparse
from zipfile import ZipFile

from ssepoptim.dataset import (
    SpeechSeparationDataset,
    SpeechSeparationDatasetConfig,
    SpeechSeparationDatasetFactory,
)
from ssepoptim.datasets.utils.audio_files_dataset import AudioFilesDataset
from ssepoptim.datasets.utils.gdrive_download import gdrive_download
from ssepoptim.datasets.utils.split_data import split_data
from ssepoptim.utils.type_checker import check_config_entries


class LibriCSSDatasetConfig(SpeechSeparationDatasetConfig):
    path: str
    sample_rate: int


class LibriCSSDataset(SpeechSeparationDataset):
    _DATASET_TYPE = SpeechSeparationDataset._DATASET_TYPE

    def __init__(self, config: LibriCSSDatasetConfig):
        self._config = config

    def _get_files(self) -> list[tuple[str, str]]:
        dataset_path = os.path.join(self._config["path"], "for_release")
        directories = ("0L", "0S", "OV10", "OV20", "OV30", "OV40")
        audio_files: list[tuple[str, str]] = []
        for directory in directories:
            directory_path = os.path.join(dataset_path, directory)
            for folder in os.listdir(directory_path):
                folder_clean_path = os.path.join(directory_path, folder, "clean")
                audio_files.append(
                    (
                        os.path.join(folder_clean_path, "mix.wav"),
                        os.path.join(folder_clean_path, "each_spk.wav"),
                    )
                )
        return audio_files

    def _get(self, subfolder: Literal["tr", "cv", "tt"]) -> _DATASET_TYPE:
        audio_files = self._get_files()
        tr, cv, tt = split_data(audio_files, [0.7, 0.2, 0.1])
        match subfolder:
            case "tr":
                dataset_files = tr
            case "cv":
                dataset_files = cv
            case "tt":
                dataset_files = tt
        return AudioFilesDataset(dataset_files, self._config["sample_rate"])

    def get_train(self) -> _DATASET_TYPE:
        return self._get("tr")

    def get_valid(self) -> _DATASET_TYPE:
        return self._get("cv")

    def get_test(self) -> _DATASET_TYPE:
        return self._get("tt")


class LibriCSSDatasetFactory(SpeechSeparationDatasetFactory):
    @staticmethod
    def _download(folder_path: str) -> None:
        def _progress_hook(block_number: int, block_size: int, total_size: int):
            total_blocks = total_size // block_size
            print(
                "\rLibriCSS downloading... {:05.2f}% [{}/{}]".format(
                    (block_number / total_blocks) * 100, block_number, total_blocks
                ),
                end=" ",
            )

        DATASET_URL = (
            "https://drive.google.com/file/d/1Piioxd5G_85K9Bhcr8ebdhXx0CnaHy7l"
        )

        file_id = urlparse(DATASET_URL).path.rsplit("/")[-1]
        zip_path = os.path.join(folder_path, "LibriCSS.zip")

        gdrive_download(file_id, zip_path, _progress_hook)

        print("\nExtracting files...")

        dataset_path = os.path.join(folder_path, "LibriCSS")

        with ZipFile(zip_path, mode="r") as zf:
            zf.extractall(dataset_path)

        os.remove(zip_path)

    @staticmethod
    def _get_config():
        return LibriCSSDatasetConfig

    @staticmethod
    def _get_object(config: SpeechSeparationDatasetConfig):
        return LibriCSSDataset(check_config_entries(config, LibriCSSDatasetConfig))
