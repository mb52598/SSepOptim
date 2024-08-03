import torch
import torchaudio

from ssepoptim.dataset import LenDataset


class AudioFilesDataset(LenDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, files: list[tuple[str, str]], expected_sampling_rate: int):
        self._files = files
        self._expected_sampling_rate = expected_sampling_rate

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_file, output_file = self._files[idx]
        input_waveform, input_sample_rate = torchaudio.load(input_file)
        output_waveform, output_sample_rate = torchaudio.load(output_file)
        if input_sample_rate != output_sample_rate:
            raise RuntimeError(
                f"Sampling rate mismatch: {input_sample_rate} and {output_sample_rate}"
            )
        if input_sample_rate != self._expected_sampling_rate:
            raise RuntimeError(
                f"Received invalid sampling rate {input_sample_rate}, expected {self._expected_sampling_rate}"
            )
        return input_waveform, output_waveform
