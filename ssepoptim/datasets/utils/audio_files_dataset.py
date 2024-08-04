import math

import torch
import torchaudio

from ssepoptim.dataset import LenDataset
from ssepoptim.utils.conversion import flatten


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


def _split_dataset_frames(
    files: list[tuple[str, str]], num_frames_per_datapoint: int
) -> list[list[tuple[int, int]]]:
    frames_start_length: list[list[tuple[int, int]]] = []
    for input_file, _ in files:
        input_info = torchaudio.info(input_file)
        datapoints = math.ceil(input_info.num_frames / num_frames_per_datapoint)
        start_length: list[tuple[int, int]] = []
        start = 0
        for _ in range(datapoints):
            start_length.append((start, num_frames_per_datapoint))
            start += num_frames_per_datapoint
        frames_start_length.append(start_length)
    return frames_start_length


class AudioFilesDataset(LenDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        files: list[tuple[str, str]],
        expected_sampling_rate: int,
        device: str | None = None,
    ):
        self._files = files
        self._device = device
        #_check_dataset(files, expected_sampling_rate)

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_file, output_file = self._files[idx]
        input_waveform, _ = torchaudio.load(input_file)
        output_waveform, _ = torchaudio.load(output_file)
        return input_waveform.to(self._device), output_waveform.to(self._device)


class SplitAudioFilesDataset(LenDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        files: list[tuple[str, str]],
        num_frames_per_datapoint: int,
        expected_sampling_rate: int,
        device: str | None = None,
    ):
        self._files = files
        self._device = device
        #_check_dataset(files, expected_sampling_rate)
        frames_start_length = _split_dataset_frames(files, num_frames_per_datapoint)
        frames_file_idx: list[int] = []
        total_datapoints = 0
        for i, lst in enumerate(frames_start_length):
            frames_file_idx.extend([i] * len(lst))
            total_datapoints += len(lst)
        self._frames_start_length = flatten(frames_start_length)
        self._frames_file_idx = frames_file_idx
        self._total_datapoints = total_datapoints

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
        return input_waveform.to(self._device), output_waveform.to(self._device)
