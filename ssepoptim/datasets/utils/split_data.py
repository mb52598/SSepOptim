from math import isclose
from typing import TypeVar

T = TypeVar("T")


def split_data(data: list[T], split_perc: list[float]) -> list[list[T]]:
    if not isclose(sum(split_perc), 1.0):
        raise RuntimeError(
            "Invalid split percentage given as it doesn't sum up to 100%"
        )
    split_lengths = [int(len(data) * perc) for perc in split_perc]
    split_lengths[0] += len(data) - sum(split_lengths)
    result: list[list[T]] = []
    start = 0
    for split_length in split_lengths:
        my_data = data[start:start + split_length]
        result.append(my_data)
        start += split_length
    return result
