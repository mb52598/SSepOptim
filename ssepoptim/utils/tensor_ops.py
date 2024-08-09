import torch

from ssepoptim.utils.conversion import fold, without_index


def _matricize_recursion(
    input: torch.Tensor,
    output: torch.Tensor,
    n: int,
    shape_index: int,
    indexes: list[int],
):
    if shape_index == input.dim():
        j = sum(
            [
                indexes[k]
                * fold(without_index(input.shape[:k], n), lambda x, y: x * y, 1)
                for k in range(input.dim())
                if k != n
            ]
        )
        output[indexes[n], j] = input[*indexes]
    else:
        for i in range(input.shape[shape_index]):
            indexes[shape_index] = i
            _matricize_recursion(input, output, n, shape_index + 1, indexes)


def matricize(tensor: torch.Tensor, n: int) -> torch.Tensor:
    dim1 = tensor.shape[n]
    dim2 = fold(without_index(tensor.shape, n), lambda x, y: x * y, 1)
    result = torch.empty(dim1, dim2)
    _matricize_recursion(tensor, result, n, 0, [0] * tensor.dim())
    return result


def _n_mode_product_recursion(
    tensor: torch.Tensor,
    matrix: torch.Tensor,
    result: torch.Tensor,
    n: int,
    shape_index: int,
    indexes: list[int],
):
    if shape_index == result.dim():
        j = indexes[n]
        summation = 0.0
        for i in range(tensor.shape[n]):
            indexes[n] = i
            summation += tensor[*indexes] * matrix[j, i]
        indexes[n] = j
        result[*indexes] = summation
    else:
        for i in range(result.shape[shape_index]):
            indexes[shape_index] = i
            _n_mode_product_recursion(
                tensor, matrix, result, n, shape_index + 1, indexes
            )


def n_mode_product(tensor: torch.Tensor, matrix: torch.Tensor, n: int) -> torch.Tensor:
    assert matrix.dim() == 2
    result = torch.empty(*tensor.shape[:n], matrix.shape[0], *tensor.shape[n + 1 :])
    _n_mode_product_recursion(tensor, matrix, result, n, 0, [0] * result.dim())
    return result


def khatri_rao_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 2
    assert b.dim() == 2
    assert a.size(1) == b.size(1)
    return torch.stack([torch.kron(a[:, k], b[:, k]) for k in range(a.size(1))], dim=-1)
