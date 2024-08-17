from functools import partial
from typing import Any, Literal, cast

import torch.nn as nn

from ssepoptim.optimization import (
    Optimization,
    OptimizationConfig,
    OptimizationFactory,
    OptimizationStage,
)
from ssepoptim.optimizations.low_rank_factorization_impl.cp import cp_als
from ssepoptim.optimizations.low_rank_factorization_impl.tucker import (
    tucker_hooi,
    tucker_hosvd,
)
from ssepoptim.utils.module_transforming import replace_module
from ssepoptim.utils.type_checker import check_config_entries


class LowRankFactorizationOptimizationConfig(OptimizationConfig):
    method: Literal["CP", "Tucker-HOSVD", "Tucker-HOOI"]
    ranks: list[int]
    num_iters: int


def _cp_conv_1d(layer: nn.Conv1d, config: LowRankFactorizationOptimizationConfig):
    if len(config["ranks"]) != 1:
        raise RuntimeError("CP decomposition can only have one rank")

    weight = layer.weight.detach()
    out_ch, in_ch, length = cp_als(weight, config["ranks"][0], config["num_iters"])

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

    return nn.Sequential(in_ch_to_length, length_to_rank, rank_to_result)


def _cp_conv_transpose_1d(
    layer: nn.ConvTranspose1d, config: LowRankFactorizationOptimizationConfig
):
    if len(config["ranks"]) != 1:
        raise RuntimeError("CP decomposition can only have one rank")

    weight = layer.weight.detach()
    in_ch, out_ch, length = cp_als(weight, config["ranks"][0], config["num_iters"])

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

    return nn.Sequential(in_ch_to_length, length_to_rank, rank_to_result)


def _cp_conv_2d(layer: nn.Conv2d, config: LowRankFactorizationOptimizationConfig):
    if len(config["ranks"]) != 1:
        raise RuntimeError("CP decomposition can only have one rank")

    weight = layer.weight.detach()
    out_ch, in_ch, vertical, horizontal = cp_als(
        weight, config["ranks"][0], config["num_iters"]
    )

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


def _cp_conv_transpose_2d(
    layer: nn.ConvTranspose2d, config: LowRankFactorizationOptimizationConfig
):
    if len(config["ranks"]) != 1:
        raise RuntimeError("CP decomposition can only have one rank")

    weight = layer.weight.detach()
    in_ch, out_ch, vertical, horizontal = cp_als(
        weight, config["ranks"][0], config["num_iters"]
    )

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


def _tucker_conv(
    layer: nn.Conv1d | nn.Conv2d | nn.Conv3d,
    config: LowRankFactorizationOptimizationConfig,
    method: Literal["HOSVD", "HOOI"],
):
    if len(config["ranks"]) != 2:
        raise RuntimeError("Partial tucker decomposition must have two ranks")

    weight = layer.weight.detach()
    match method:
        case "HOSVD":
            core, (out_ch, in_ch) = tucker_hosvd(weight, config["ranks"])
        case "HOOI":
            core, (out_ch, in_ch) = tucker_hooi(
                weight, config["ranks"], config["num_iters"]
            )

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


def _tucker_conv_transpose(
    layer: nn.ConvTranspose1d | nn.ConvTranspose2d | nn.ConvTranspose3d,
    config: LowRankFactorizationOptimizationConfig,
    method: Literal["HOSVD", "HOOI"],
):
    if len(config["ranks"]) != 2:
        raise RuntimeError("Partial tucker decomposition must have two ranks")

    weight = layer.weight.detach()
    match method:
        case "HOSVD":
            core, (in_ch, out_ch) = tucker_hosvd(weight, config["ranks"])
        case "HOOI":
            core, (in_ch, out_ch) = tucker_hooi(
                weight, config["ranks"], config["num_iters"]
            )

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


class LowRankFactorizationOptimization(Optimization):
    def __init__(self, config: LowRankFactorizationOptimizationConfig):
        self._config = config
        self._method_mapping = {
            "CP": {
                nn.Conv1d: _cp_conv_1d,
                nn.Conv2d: _cp_conv_2d,
                nn.ConvTranspose1d: _cp_conv_transpose_1d,
                nn.ConvTranspose2d: _cp_conv_transpose_2d,
            },
            "Tucker-HOSVD": {
                nn.Conv1d: partial(_tucker_conv, method="HOSVD"),
                nn.Conv2d: partial(_tucker_conv, method="HOSVD"),
                nn.Conv3d: partial(_tucker_conv, method="HOSVD"),
                nn.ConvTranspose1d: partial(_tucker_conv_transpose, method="HOSVD"),
                nn.ConvTranspose2d: partial(_tucker_conv_transpose, method="HOSVD"),
                nn.ConvTranspose3d: partial(_tucker_conv_transpose, method="HOSVD"),
            },
            "Tucker-HOOI": {
                nn.Conv1d: partial(_tucker_conv, method="HOOI"),
                nn.Conv2d: partial(_tucker_conv, method="HOOI"),
                nn.Conv3d: partial(_tucker_conv, method="HOOI"),
                nn.ConvTranspose1d: partial(_tucker_conv_transpose, method="HOSVD"),
                nn.ConvTranspose2d: partial(_tucker_conv_transpose, method="HOSVD"),
                nn.ConvTranspose3d: partial(_tucker_conv_transpose, method="HOSVD"),
            },
        }
        self._finetune_done = False

    def apply(
        self, module: nn.Module, stage: OptimizationStage, locals: dict[str, Any]
    ) -> nn.Module:
        if stage != "FINETUNE_START":
            return module
        # Replace convolutional layers based on method
        for layer, function in self._method_mapping[self._config["method"]].items():
            replace_module(
                module, layer, lambda x: function(cast(Any, x), self._config)
            )
        # Set flag
        self._finetune_done = True
        # Return module
        return module

    def requiresFinetune(self) -> bool:
        return not self._finetune_done


class LowRankFactorizationOptimizationFactory(OptimizationFactory):
    @staticmethod
    def _get_config():
        return LowRankFactorizationOptimizationConfig

    @staticmethod
    def _get_object(config: OptimizationConfig):
        return LowRankFactorizationOptimization(
            check_config_entries(config, LowRankFactorizationOptimizationConfig)
        )
