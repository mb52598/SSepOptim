import os
from typing import Any

import pandas as pd
import torch
import torchaudio

from ssepoptim.dataset import LenDataset


class CsvAudioDataset(LenDataset[tuple[torch.Tensor, int]]):
    def __init__(self, csv_path: str, filename_column: str, data_path: str):
        self._df: pd.Series[Any] = pd.read_csv(csv_path).loc[:, filename_column]
        self._data_path = data_path

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path = os.path.join(
            self._data_path,
            self._df.iat[idx],
        )
        with open(path, "rb") as file:
            waveform, sample_rate = torchaudio.load(file)
            return waveform, sample_rate
