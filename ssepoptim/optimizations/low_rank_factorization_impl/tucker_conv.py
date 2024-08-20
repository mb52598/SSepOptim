from typing import Literal

import torch.nn as nn

from ssepoptim.optimizations.low_rank_factorization_impl.tucker import (
    tucker_hooi,
    tucker_hosvd,
)


def ptucker_conv(
    layer: nn.Conv1d | nn.Conv2d | nn.Conv3d,
    ranks: tuple[int, int],
    num_iters: int,
    method: Literal["HOSVD", "HOOI"],
):
    weight = layer.weight.detach()
    match method:
        case "HOSVD":
            core, (out_ch, in_ch) = tucker_hosvd(weight, list(ranks))
        case "HOOI":
            core, (out_ch, in_ch) = tucker_hooi(weight, list(ranks), num_iters)

    nnConvClass = type(layer)
    match nnConvClass:
        case nn.Conv1d:
            n = 1
        case nn.Conv2d:
            n = 2
        case nn.Conv3d:
            n = 3
        case _:
            raise RuntimeError(f"Unsupported layer class {layer.__class__.__name__}")

    in_ch_to_length = nnConvClass(
        in_channels=in_ch.shape[0],
        out_channels=in_ch.shape[1],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=False,
    )

    length_to_final_rank = nnConvClass(
        in_channels=in_ch.shape[1],
        out_channels=out_ch.shape[1],
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        bias=False,
    )

    final_rank_to_out_ch = nnConvClass(
        in_channels=out_ch.shape[1],
        out_channels=out_ch.shape[0],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=layer.bias is not None,
    )

    if layer.bias is not None:
        final_rank_to_out_ch.bias.data = layer.bias.data

    in_ch = in_ch.T
    for _ in range(n):
        in_ch = in_ch.unsqueeze(-1)
        out_ch = out_ch.unsqueeze(-1)

    in_ch_to_length.weight.data = in_ch
    length_to_final_rank.weight.data = core
    final_rank_to_out_ch.weight.data = out_ch

    return nn.Sequential(in_ch_to_length, length_to_final_rank, final_rank_to_out_ch)


def ptucker_conv_transpose(
    layer: nn.ConvTranspose1d | nn.ConvTranspose2d | nn.ConvTranspose3d,
    ranks: tuple[int, int],
    num_iters: int,
    method: Literal["HOSVD", "HOOI"],
):
    weight = layer.weight.detach()
    match method:
        case "HOSVD":
            core, (in_ch, out_ch) = tucker_hosvd(weight, list(ranks))
        case "HOOI":
            core, (in_ch, out_ch) = tucker_hooi(weight, list(ranks), num_iters)

    nnConvClass = type(layer)
    match nnConvClass:
        case nn.ConvTranspose1d:
            n = 1
        case nn.ConvTranspose2d:
            n = 2
        case nn.ConvTranspose3d:
            n = 3
        case _:
            raise RuntimeError(f"Unsupported layer class {layer.__class__.__name__}")

    in_ch_to_length = nnConvClass(
        in_channels=in_ch.shape[0],
        out_channels=in_ch.shape[1],
        kernel_size=1,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=layer.dilation,
        bias=False,
    )

    length_to_final_rank = nnConvClass(
        in_channels=in_ch.shape[1],
        out_channels=out_ch.shape[1],
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        output_padding=layer.output_padding,
        dilation=layer.dilation,
        bias=False,
    )

    final_rank_to_out_ch = nnConvClass(
        in_channels=out_ch.shape[1],
        out_channels=out_ch.shape[0],
        kernel_size=1,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=layer.dilation,
        bias=layer.bias is not None,
    )

    if layer.bias is not None:
        final_rank_to_out_ch.bias.data = layer.bias.data

    out_ch = out_ch.T
    for _ in range(n):
        in_ch = in_ch.unsqueeze(-1)
        out_ch = out_ch.unsqueeze(-1)

    in_ch_to_length.weight.data = in_ch
    length_to_final_rank.weight.data = core
    final_rank_to_out_ch.weight.data = out_ch

    return nn.Sequential(in_ch_to_length, length_to_final_rank, final_rank_to_out_ch)
