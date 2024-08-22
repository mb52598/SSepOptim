import torch.nn as nn

from ssepoptim.optimizations.low_rank_factorization_impl.cp import cp_als


def cp_conv_1d(layer: nn.Conv1d, rank: int, num_iters: int):
    weight = layer.weight.detach()
    out_ch, in_ch, length = cp_als(weight, rank, num_iters)

    in_ch_to_length = nn.Conv1d(
        in_channels=in_ch.shape[0],
        out_channels=in_ch.shape[1],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=False,
    )

    length_to_rank = nn.Conv1d(
        in_channels=length.shape[1],
        out_channels=1,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        bias=False,
    )

    rank_to_result = nn.Conv1d(
        in_channels=1,
        out_channels=out_ch.shape[0],
        kernel_size=out_ch.shape[1],
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=layer.bias is not None,
    )

    if layer.bias is not None:
        rank_to_result.bias.data = layer.bias.data

    in_ch_to_length.weight.data = in_ch.T.unsqueeze(-1)
    length_to_rank.weight.data = length.T.unsqueeze(0)
    rank_to_result.weight.data = out_ch.unsqueeze(1)

    return nn.Sequential(
        in_ch_to_length,
        length_to_rank,
        rank_to_result,
        nn.ZeroPad1d(padding=(0, out_ch.shape[1] - 1)),
    )


def cp_conv_transpose_1d(layer: nn.ConvTranspose1d, rank: int, num_iters: int):
    weight = layer.weight.detach()
    in_ch, out_ch, length = cp_als(weight, rank, num_iters)

    in_ch_to_length = nn.ConvTranspose1d(
        in_channels=in_ch.shape[0],
        out_channels=in_ch.shape[1],
        kernel_size=1,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=layer.dilation,
        bias=False,
    )

    length_to_rank = nn.ConvTranspose1d(
        in_channels=length.shape[1],
        out_channels=1,
        kernel_size=length.shape[0],
        stride=layer.stride,
        padding=layer.padding,
        output_padding=layer.output_padding,
        dilation=layer.dilation,
        bias=False,
    )

    rank_to_result = nn.ConvTranspose1d(
        in_channels=1,
        out_channels=out_ch.shape[0],
        kernel_size=out_ch.shape[1],
        stride=1,
        padding=0,
        output_padding=0,
        dilation=layer.dilation,
        bias=layer.bias is not None,
    )

    if layer.bias is not None:
        rank_to_result.bias.data = layer.bias.data

    in_ch_to_length.weight.data = in_ch.unsqueeze(-1)
    length_to_rank.weight.data = length.T.unsqueeze(1)
    rank_to_result.weight.data = out_ch.unsqueeze(0)

    return nn.Sequential(
        in_ch_to_length,
        length_to_rank,
        rank_to_result,
        nn.ZeroPad1d(padding=(0, out_ch.shape[1] - 1)),
    )


def cp_conv_2d(layer: nn.Conv2d, rank: int, num_iters: int):
    weight = layer.weight.detach()
    out_ch, in_ch, vertical, horizontal = cp_als(weight, rank, num_iters)

    in_ch_to_horizontal = nn.Conv2d(
        in_channels=in_ch.shape[0],
        out_channels=in_ch.shape[1],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=False,
    )

    horizontal_to_vertical = nn.Conv2d(
        in_channels=horizontal.shape[1],
        out_channels=1,
        kernel_size=(1, horizontal.shape[0]),
        stride=(1, layer.stride[1]),
        padding=(0, layer.padding[1]),
        dilation=layer.dilation,
        bias=False,
    )

    vertical_to_rank = nn.Conv2d(
        in_channels=1,
        out_channels=vertical.shape[1],
        kernel_size=(vertical.shape[0], 1),
        stride=(layer.stride[0], 1),
        padding=(layer.padding[0], 0),
        dilation=layer.dilation,
        bias=False,
    )

    rank_to_result = nn.Conv2d(
        in_channels=out_ch.shape[1],
        out_channels=out_ch.shape[0],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=layer.bias is not None,
    )

    if layer.bias is not None:
        rank_to_result.bias.data = layer.bias.data

    in_ch_to_horizontal.weight.data = in_ch.T.unsqueeze(-1).unsqueeze(-1)
    horizontal_to_vertical.weight.data = horizontal.T.unsqueeze(1).unsqueeze(0)
    vertical_to_rank.weight.data = vertical.T.unsqueeze(1).unsqueeze(-1)
    rank_to_result.weight.data = out_ch.unsqueeze(-1).unsqueeze(-1)

    return nn.Sequential(
        in_ch_to_horizontal,
        horizontal_to_vertical,
        vertical_to_rank,
        rank_to_result,
    )


def cp_conv_transpose_2d(layer: nn.ConvTranspose2d, rank: int, num_iters: int):
    weight = layer.weight.detach()
    in_ch, out_ch, vertical, horizontal = cp_als(weight, rank, num_iters)

    in_ch_to_horizontal = nn.ConvTranspose2d(
        in_channels=in_ch.shape[0],
        out_channels=in_ch.shape[1],
        kernel_size=1,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=layer.dilation,
        bias=False,
    )

    horizontal_to_vertical = nn.ConvTranspose2d(
        in_channels=horizontal.shape[1],
        out_channels=1,
        kernel_size=(1, horizontal.shape[0]),
        stride=(1, layer.stride[1]),
        padding=(0, layer.padding[1]),
        output_padding=(0, layer.output_padding[1]),
        dilation=layer.dilation,
        bias=False,
    )

    vertical_to_rank = nn.ConvTranspose2d(
        in_channels=1,
        out_channels=vertical.shape[1],
        kernel_size=(vertical.shape[0], 1),
        stride=(layer.stride[0], 1),
        padding=(layer.padding[0], 0),
        output_padding=(layer.output_padding[0], 0),
        dilation=layer.dilation,
        bias=False,
    )

    rank_to_result = nn.ConvTranspose2d(
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
        rank_to_result.bias.data = layer.bias.data

    in_ch_to_horizontal.weight.data = in_ch.unsqueeze(-1).unsqueeze(-1)
    horizontal_to_vertical.weight.data = horizontal.T.unsqueeze(-2).unsqueeze(-2)
    vertical_to_rank.weight.data = vertical.T.unsqueeze(0).unsqueeze(-1)
    rank_to_result.weight.data = out_ch.T.unsqueeze(-1).unsqueeze(-1)

    return nn.Sequential(
        in_ch_to_horizontal,
        horizontal_to_vertical,
        vertical_to_rank,
        rank_to_result,
    )
