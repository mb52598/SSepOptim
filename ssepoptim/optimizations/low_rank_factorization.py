from functools import partial
from typing import Any, Literal, cast

import torch.nn as nn

from ssepoptim.optimization import (
    Optimization,
    OptimizationConfig,
    OptimizationFactory,
    OptimizationStage,
)
from ssepoptim.optimizations.low_rank_factorization_impl.cp import (
    cp_estimate_rank,
    cp_number_of_parameters,
)
from ssepoptim.optimizations.low_rank_factorization_impl.cp_conv import (
    cp_conv_1d,
    cp_conv_2d,
    cp_conv_transpose_1d,
    cp_conv_transpose_2d,
)
from ssepoptim.optimizations.low_rank_factorization_impl.tucker import (
    ptucker_estimate_ranks,
    ptucker_number_of_parameters,
)
from ssepoptim.optimizations.low_rank_factorization_impl.tucker_conv import (
    ptucker_conv,
    ptucker_conv_transpose,
)
from ssepoptim.utils.chaining import chain_two_values
from ssepoptim.utils.module_transforming import replace_module
from ssepoptim.utils.type_checker import check_config_entries


class LowRankFactorizationOptimizationConfig(OptimizationConfig):
    method: Literal["CP", "Tucker-HOSVD", "Tucker-HOOI"]
    keep_percentage: float
    num_iters: int


def _cp_conv(
    layer: nn.Conv1d | nn.Conv2d | nn.ConvTranspose1d | nn.ConvTranspose2d,
    config: LowRankFactorizationOptimizationConfig,
):
    layer_type = type(layer)
    match layer_type:
        case nn.Conv1d:
            func = cp_conv_1d
        case nn.Conv2d:
            func = cp_conv_2d
        case nn.ConvTranspose1d:
            func = cp_conv_transpose_1d
        case nn.ConvTranspose2d:
            func = cp_conv_transpose_2d
        case _:
            raise RuntimeError(f"Unsupported layer type {layer_type.__name__}")
    shape = layer.weight.shape
    rank = cp_estimate_rank(shape, config["keep_percentage"])
    # If our decomposition has more parameters, then just use the initial layer
    if cp_number_of_parameters(shape, rank) >= chain_two_values(
        shape, lambda x, y: x * y
    ):
        return layer
    # Otherwise return decomposition
    return func(cast(Any, layer), rank, config["num_iters"])


def _ptucker_conv(
    layer: nn.Conv1d
    | nn.Conv2d
    | nn.Conv3d
    | nn.ConvTranspose1d
    | nn.ConvTranspose2d
    | nn.ConvTranspose3d,
    config: LowRankFactorizationOptimizationConfig,
    method: Literal["HOSVD", "HOOI"],
):
    layer_type = type(layer)
    match layer_type:
        case nn.Conv1d | nn.Conv2d | nn.Conv3d:
            func = ptucker_conv
        case nn.ConvTranspose1d | nn.ConvTranspose2d | nn.ConvTranspose3d:
            func = ptucker_conv_transpose
        case _:
            raise RuntimeError(f"Unsupported layer type {layer_type.__name__}")
    shape = layer.weight.shape
    ranks = ptucker_estimate_ranks(shape, config["keep_percentage"])
    # If our decomposition has more parameters, then just use the initial layer
    if ptucker_number_of_parameters(shape, ranks) >= chain_two_values(
        shape, lambda x, y: x * y
    ):
        return layer
    # Otherwise return decomposition
    return func(cast(Any, layer), ranks, config["num_iters"], method)


class LowRankFactorizationOptimization(Optimization):
    def __init__(self, config: LowRankFactorizationOptimizationConfig):
        self._config = config
        self._method_mapping = {
            "CP": {
                nn.Conv1d: _cp_conv,
                nn.Conv2d: _cp_conv,
                nn.ConvTranspose1d: _cp_conv,
                nn.ConvTranspose2d: _cp_conv,
            },
            "Tucker-HOSVD": {
                nn.Conv1d: partial(_ptucker_conv, method="HOSVD"),
                nn.Conv2d: partial(_ptucker_conv, method="HOSVD"),
                nn.Conv3d: partial(_ptucker_conv, method="HOSVD"),
                nn.ConvTranspose1d: partial(_ptucker_conv, method="HOSVD"),
                nn.ConvTranspose2d: partial(_ptucker_conv, method="HOSVD"),
                nn.ConvTranspose3d: partial(_ptucker_conv, method="HOSVD"),
            },
            "Tucker-HOOI": {
                nn.Conv1d: partial(_ptucker_conv, method="HOOI"),
                nn.Conv2d: partial(_ptucker_conv, method="HOOI"),
                nn.Conv3d: partial(_ptucker_conv, method="HOOI"),
                nn.ConvTranspose1d: partial(_ptucker_conv, method="HOOI"),
                nn.ConvTranspose2d: partial(_ptucker_conv, method="HOOI"),
                nn.ConvTranspose3d: partial(_ptucker_conv, method="HOOI"),
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
