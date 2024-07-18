from types import TracebackType
from typing import Optional, Self, Type

import torch
import torch.nn as nn

from ssepoptim.optimization import (
    Optimization,
    OptimizationConfig,
    OptimizationContext,
    OptimizationFactory,
)
from ssepoptim.utils.checker import check_config_entries


class QuantizationOptimizationConfig(OptimizationConfig):
    device: str
    dtype: str


class QuantizationOptimizationContext:
    def __init__(self, config: QuantizationOptimizationConfig):
        dtype = getattr(torch, config["dtype"])
        if type(dtype) is not torch.dtype:
            raise RuntimeError(
                'Invalid quantization type "{}"'.format(type(dtype).__name__)
            )
        self._autocast = torch.autocast(device_type=config["device"], dtype=dtype)

    def __enter__(self) -> Self:
        self._autocast.__enter__()
        return self

    def __exit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> None:
        self._autocast.__exit__(exctype, excinst, exctb)


class QuantizationOptimization(Optimization):
    def __init__(self, config: QuantizationOptimizationConfig):
        self._config = config

    def apply(self, model: nn.Module) -> OptimizationContext:
        return QuantizationOptimizationContext(self._config)


class QuantizationOptimizationFactory(OptimizationFactory):
    @staticmethod
    def _get_config():
        return QuantizationOptimizationConfig

    @staticmethod
    def _get_object(config: OptimizationConfig):
        return QuantizationOptimization(
            check_config_entries(config, QuantizationOptimizationConfig)
        )
