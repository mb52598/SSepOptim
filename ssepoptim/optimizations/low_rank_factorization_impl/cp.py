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
