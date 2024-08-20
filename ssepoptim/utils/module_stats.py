from typing import Iterator

import torch.nn as nn


def get_module_parameters(
    module: nn.Module, only_trainable: bool = False
) -> Iterator[nn.Parameter]:
    return (
        parameter
        for parameter in module.parameters()
        if not only_trainable or parameter.requires_grad
    )


def get_module_named_parameters(
    module: nn.Module, only_trainable: bool = False
) -> Iterator[tuple[str, nn.Parameter]]:
    return (
        (name, parameter)
        for name, parameter in module.named_parameters()
        if not only_trainable or parameter.requires_grad
    )


def get_module_parameters_count(module: nn.Module, only_trainable: bool = False) -> int:
    return sum(
        parameter.numel() for parameter in get_module_parameters(module, only_trainable)
    )


def get_module_parameters_memory_usage(
    module: nn.Module, only_trainable: bool = False
) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in get_module_parameters(module, only_trainable)
    )


def get_module_named_parameters_count(
    module: nn.Module, only_trainable: bool = False
) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    for name, parameter in get_module_named_parameters(module, only_trainable):
        result.append((name, parameter.numel()))
    return result


def get_module_named_parameters_memory_usage(
    module: nn.Module, only_trainable: bool = False
) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    for name, parameter in get_module_named_parameters(module, only_trainable):
        result.append((name, parameter.numel() * parameter.element_size()))
    return result
