import torch

from ssepoptim.utils.tensor_ops import matricize


def leading_left_singular_vectors(x: torch.Tensor, n: int, rank: int) -> torch.Tensor:
    return torch.svd(matricize(x, n)).U[:, :rank]


def leading_left_singular_vectors_with_random(
    x: torch.Tensor, n: int, rank: int
) -> torch.Tensor:
    value = leading_left_singular_vectors(x, n, rank)
    if value.shape != (x.shape[n], rank):
        value = torch.randn(x.shape[n], rank)
    return value


def init_matrices(x: torch.Tensor, ranks: list[int]) -> list[torch.Tensor]:
    return [
        leading_left_singular_vectors_with_random(x, i, rank)
        for i, rank in enumerate(ranks)
    ]
