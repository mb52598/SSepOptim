import math
from typing import TypeVar

import torchaudio

from ssepoptim.utils.conversion import flatten

T = TypeVar("T")


def split_data(data: list[T], split_perc: list[float]) -> list[list[T]]:
    if not math.isclose(sum(split_perc), 1.0):
        raise RuntimeError(
            "Invalid split percentage given as it doesn't sum up to 100%"
        )
    split_lengths = [int(len(data) * perc) for perc in split_perc]
    split_lengths[0] += len(data) - sum(split_lengths)
    result: list[list[T]] = []
    start = 0
    for split_length in split_lengths:
        my_data = data[start : start + split_length]
        result.append(my_data)
        start += split_length
    return result


def split_dataset_frames(
    files: list[str], num_frames_per_datapoint: int
) -> list[list[tuple[int, int]]]:
    frames_start_length: list[list[tuple[int, int]]] = []
    for file in files:
        # Get audio info for the number of frames
        info = torchaudio.info(file)
        # Calculate how many num_frames_per_datapoint segments we have in one file
        # We can use ceil because we will be reading the file with torchaudio.load
        #    which ignores if the length parameter is bigger than the audio length
        datapoints = math.ceil(info.num_frames / num_frames_per_datapoint)
        # Calculate (start, length) pairs for the current file
        start_length: list[tuple[int, int]] = []
        start = 0
        for _ in range(datapoints):
            start_length.append((start, num_frames_per_datapoint))
            start += num_frames_per_datapoint
        # Add the current file split info to the list of all files
        frames_start_length.append(start_length)
    return frames_start_length


def split_dataset_frames_idx_size(
    files: list[str], num_frames_per_datapoint: int
) -> tuple[list[tuple[int, int]], list[int], int]:
    # Get files (start, length) pairs
    frames_start_length = split_dataset_frames(files, num_frames_per_datapoint)
    # Create a list which will keep tract which file number is at which index
    frames_file_idx: list[int] = []
    # Number to count total datapoints
    total_datapoints = 0
    for i, lst in enumerate(frames_start_length):
        # len(lst) is the number of datapoints in the current file
        frames_file_idx.extend([i] * len(lst))
        total_datapoints += len(lst)
    # We can now flatten frames_start_length as frame_file_idx keeps track of the indexes
    return flatten(frames_start_length), frames_file_idx, total_datapoints
