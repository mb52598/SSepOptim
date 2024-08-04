import pandas as pd
import torch
import torchaudio

from ssepoptim.dataset import LenDataset


class CsvAudioDataset(LenDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        csv_path: str,
        mixture_field: str,
        source_fields: list[str],
        device: str | None = None,
    ):
        self._df = pd.read_csv(csv_path)
        self._mixture_field_idx = self._df.columns.get_loc(mixture_field)
        self._source_field_idxs = [
            self._df.columns.get_loc(source_field) for source_field in source_fields
        ]
        self._device = device

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
        return mixture_tensor.to(self._device), source_tensor.to(self._device)
