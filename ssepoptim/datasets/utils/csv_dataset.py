import pandas as pd
import torch
import torchaudio

from ssepoptim.dataset import LenDataset
from ssepoptim.datasets.utils.split_data import split_dataset_frames_idx_size


class CsvAudioDataset(LenDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        csv_path: str,
        mixture_field: str,
        source_fields: list[str],
    ):
        self._df = pd.read_csv(csv_path)
        self._mixture_field_idx = self._df.columns.get_loc(mixture_field)
        self._source_field_idxs = [
            self._df.columns.get_loc(source_field) for source_field in source_fields
        ]

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        mixture_file: str = self._df.iat[idx, self._mixture_field_idx]
        source_files: list[str] = [
            self._df.iat[idx, source_field_idx]
            for source_field_idx in self._source_field_idxs
        ]
        mixture_tensor, _ = torchaudio.load(mixture_file)
        source_tensor = torch.concat(
            [torchaudio.load(source_file)[0] for source_file in source_files]
        )
        return mixture_tensor, source_tensor


class SplitCsvAudioDataset(LenDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        csv_path: str,
        mixture_field: str,
        source_fields: list[str],
        num_frames_per_datapoint: int,
    ):
        self._df = pd.read_csv(csv_path)
        self._mixture_field_idx = self._df.columns.get_loc(mixture_field)
        self._source_field_idxs = [
            self._df.columns.get_loc(source_field) for source_field in source_fields
        ]
        self._frames_start_length, self._frames_file_idx, self._total_datapoints = (
            split_dataset_frames_idx_size(
                self._df.iloc[:, self._mixture_field_idx].to_list(),
                num_frames_per_datapoint,
            )
        )

    def __len__(self) -> int:
        return self._total_datapoints

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Get file start and length
        start, length = self._frames_start_length[idx]
        # Get file index
        file_idx = self._frames_file_idx[idx]
        # Get mixture file path
        mixture_file: str = self._df.iat[file_idx, self._mixture_field_idx]
        # Get source file paths
        source_files: list[str] = [
            self._df.iat[file_idx, source_field_idx]
            for source_field_idx in self._source_field_idxs
        ]
        # Load mixture file
        mixture_tensor, _ = torchaudio.load(
            mixture_file, frame_offset=start, num_frames=length
        )
        # Load source files
        source_tensor = torch.concat(
            [
                torchaudio.load(source_file, frame_offset=start, num_frames=length)[0]
                for source_file in source_files
            ]
        )
        # Return mixture and sorce tensors
        return mixture_tensor, source_tensor
