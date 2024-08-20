from typing import cast

import torch

import ssepoptim_extensions_unique_argmin


class UniqueArgmin(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, input: torch.Tensor):
        x, row, col = cast(
            list[torch.Tensor], ssepoptim_extensions_unique_argmin.forward(input)
        )
        ctx.save_for_backward(row, col)
        return x

    @staticmethod
    def backward(
        ctx: torch.autograd.function.BackwardCFunction, grad_output: torch.Tensor
    ):
        row: torch.Tensor
        col: torch.Tensor
        row, col = cast(list[torch.Tensor], ctx.saved_tensors)
        x = torch.zeros(row.size(0), col.size(0))
        x[row, col] = 1
        return x * grad_output
