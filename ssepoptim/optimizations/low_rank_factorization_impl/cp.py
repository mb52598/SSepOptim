import math
from typing import Callable

import torch

from ssepoptim.optimizations.low_rank_factorization_impl.base import init_matrices
from ssepoptim.utils.chaining import chain_two_values
from ssepoptim.utils.conversion import without_index
from ssepoptim.utils.tensor_ops import khatri_rao_product, matricize


def cp_als(x: torch.Tensor, rank: int, iters: int) -> list[torch.Tensor]:
    matrices = init_matrices(x, [rank] * x.dim())
    for _ in range(iters):
        for i in range(x.dim()):
            matrices_i = without_index(matrices, i)
            # Hadamard of RxR matrices
            v = chain_two_values(
                [matrix.T @ matrix for matrix in matrices_i], torch.mul
            )
            # Khatri–Rao
            matrices[i] = (
                matricize(x, i)
                @ chain_two_values(reversed(matrices_i), khatri_rao_product)
                @ v.pinverse()
            )
    return matrices


def cp_estimate_rank(
    weight_shape: tuple[int, ...],
    keep_percentage: float,
    round_fn: Callable[[float], int] = math.ceil,
):
    # total_estimated_values / total_noestimated_values = keep_percentage
    if len(weight_shape) < 1:
        raise RuntimeError(
            "Partial tucker rank estimation needs weight of atleast one dimension"
        )

    return round_fn(
        keep_percentage
        * (
            chain_two_values(weight_shape, lambda x, y: x * y)
            / chain_two_values(weight_shape, lambda x, y: x + y)
        )
    )


def cp_number_of_parameters(weight_shape: tuple[int, ...], rank: int) -> int:
    return sum(s * rank for s in weight_shape)
