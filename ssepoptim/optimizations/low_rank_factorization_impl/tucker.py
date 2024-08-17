import torch

from ssepoptim.optimizations.low_rank_factorization_impl.base import (
    init_matrices,
    leading_left_singular_vectors,
)
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
