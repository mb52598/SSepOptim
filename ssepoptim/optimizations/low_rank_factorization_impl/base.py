import torch

from ssepoptim.utils.tensor_ops import matricize


def leading_left_singular_vectors(x: torch.Tensor, n: int, rank: int) -> torch.Tensor:
    return torch.svd(matricize(x, n)).U[:, :rank]


def init_matrices(x: torch.Tensor, ranks: list[int]) -> list[torch.Tensor]:
    return [leading_left_singular_vectors(x, i, ranks[i]) for i in range(x.dim())]
