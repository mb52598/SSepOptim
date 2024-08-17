import torch
import torchaudio

from ssepoptim.dataset import LenDataset
from ssepoptim.datasets.utils.split_data import split_dataset_frames_idx_size


def _check_dataset(files: list[tuple[str, str]], expected_sampling_rate: int):
    for input_file, output_file in files:
        input_info = torchaudio.info(input_file)
        output_info = torchaudio.info(output_file)
        if input_info.sample_rate != output_info.sample_rate:
            raise RuntimeError(
                f"Sampling rate mismatch: {input_info.sample_rate} and {output_info.sample_rate}"
            )
        if input_info.sample_rate != expected_sampling_rate:
            raise RuntimeError(
                f"Received invalid sampling rate {input_info.sample_rate}, expected {expected_sampling_rate}"
            )
        if input_info.num_frames != output_info.num_frames:
            raise RuntimeError(
                f"Number of frames mismatch: {input_info.num_frames} and {output_info.num_frames}"
            )


class AudioFilesDataset(LenDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        files: list[tuple[str, str]],
        expected_sampling_rate: int,
    ):
        self._files = files
        # _check_dataset(files, expected_sampling_rate)

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_file, output_file = self._files[idx]
        input_waveform, _ = torchaudio.load(input_file)
        output_waveform, _ = torchaudio.load(output_file)
        return input_waveform, output_waveform


class SplitAudioFilesDataset(LenDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        files: list[tuple[str, str]],
        num_frames_per_datapoint: int,
        expected_sampling_rate: int,
    ):
        self._files = files
        # _check_dataset(files, expected_sampling_rate)
        self._frames_start_length, self._frames_file_idx, self._total_datapoints = (
            split_dataset_frames_idx_size(
                [file[0] for file in files], num_frames_per_datapoint
            )
        )

    def __len__(self) -> int:
        return self._total_datapoints

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_file, output_file = self._files[self._frames_file_idx[idx]]
        start, length = self._frames_start_length[idx]
        input_waveform, _ = torchaudio.load(
            input_file, frame_offset=start, num_frames=length
        )
        output_waveform, _ = torchaudio.load(
            output_file, frame_offset=start, num_frames=length
        )
        return input_waveform, output_waveform
