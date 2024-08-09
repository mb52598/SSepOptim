import timeit
from typing import Any, Iterator

import torch.nn as nn


def get_model_parameters(
    model: nn.Module, only_trainable: bool = False
) -> Iterator[nn.Parameter]:
    return (
        parameter
        for parameter in model.parameters()
        if not only_trainable or parameter.requires_grad
    )


def get_model_named_parameters(
    model: nn.Module, only_trainable: bool = False
) -> Iterator[tuple[str, nn.Parameter]]:
    return (
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not only_trainable or parameter.requires_grad
    )


def get_model_parameters_count(model: nn.Module, only_trainable: bool = False) -> int:
    return sum(
        parameter.numel() for parameter in get_model_parameters(model, only_trainable)
    )


def get_model_parameters_memory_usage(
    model: nn.Module, only_trainable: bool = False
) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in get_model_parameters(model, only_trainable)
    )


def get_model_named_parameters_count(
    model: nn.Module, only_trainable: bool = False
) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    for name, parameter in get_model_named_parameters(model, only_trainable):
        result.append((name, parameter.numel()))
    return result


def get_model_named_parameters_memory_usage(
    model: nn.Module, only_trainable: bool = False
) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    for name, parameter in get_model_named_parameters(model, only_trainable):
        result.append((name, parameter.numel() * parameter.element_size()))
    return result


def get_model_latency(model: nn.Module, *input: Any, number: int = 1000000) -> float:
    return timeit.timeit(lambda: model(*input), number=number)
