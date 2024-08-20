import math
from typing import Callable

import torch

from ssepoptim.optimizations.low_rank_factorization_impl.base import (
    init_matrices,
    leading_left_singular_vectors,
)
from ssepoptim.utils.chaining import chain_two_values
from ssepoptim.utils.tensor_ops import n_mode_product


def n_mode_multiply(x: torch.Tensor, matrices: list[torch.Tensor]) -> torch.Tensor:
    tensor = x
    for i, matrix in enumerate(matrices):
        tensor = n_mode_product(tensor, matrix.T, i)
    return tensor


def n_mode_multiply_without(
    x: torch.Tensor, matrices: list[torch.Tensor], index: int
) -> torch.Tensor:
    tensor = x
    for i, matrix in enumerate(matrices):
        if i == index:
            continue
        tensor = n_mode_product(tensor, matrix.T, i)
    return tensor


def tucker_hosvd(
    x: torch.Tensor, ranks: list[int]
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    matrices = init_matrices(x, ranks)
    return n_mode_multiply(x, matrices), matrices


def tucker_hooi(x: torch.Tensor, ranks: list[int], iters: int):
    matrices = init_matrices(x, ranks)
    for _ in range(iters):
        for i, rank in enumerate(ranks):
            y = n_mode_multiply_without(x, matrices, i)
            matrices[i] = leading_left_singular_vectors(y, i, rank)
    return n_mode_multiply(x, matrices), matrices


def ptucker_estimate_ranks(
    weight_shape: tuple[int, ...],
    keep_percentage: float,
    round_fn: Callable[[float], int] = math.ceil,
):
    # total_estimated_values / total_noestimated_values = keep_percentage, with r1/s1 = r2/s2
    if len(weight_shape) < 2:
        raise RuntimeError(
            "Partial tucker rank estimation needs weight of atleast two dimensions"
        )

    A = (weight_shape[0] ** 2 + weight_shape[1] ** 2) / weight_shape[0]
    B = chain_two_values(weight_shape, lambda x, y: x * y)
    C = weight_shape[0] ** 2
    NUM = (-A) + math.sqrt(A**2 + (4 * (B**2) * keep_percentage) / C)
    DENOM = (2 * B) / C
    R1 = round_fn(NUM / DENOM)
    R2 = round_fn((R1 * weight_shape[1]) / weight_shape[0])

    return R1, R2


def ptucker_number_of_parameters(
    weight_shape: tuple[int, ...], ranks: tuple[int, int]
) -> int:
    if len(weight_shape) < 2:
        raise RuntimeError(
            "Partial tucker number of parameters needs weight of atleast two dimensions"
        )
    return (
        weight_shape[0] * ranks[0]
        + weight_shape[1] * ranks[1]
        + chain_two_values([*ranks, *weight_shape[2:]], lambda x, y: x * y)
    )
