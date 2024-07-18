from types import TracebackType
from typing import Optional, Self, Type

import torch.nn as nn
import torch.nn.utils.prune as prune

from ssepoptim.optimization import (
    Optimization,
    OptimizationConfig,
    OptimizationContext,
    OptimizationFactory,
)
from ssepoptim.utils.checker import check_config_entries


class PruningOptimizationConfig(OptimizationConfig):
    pass


class PruningOptimizationContext:
    def __init__(self, model: nn.Module):
        self._model = model

    def __enter__(self) -> Self:
        prune.global_unstructured(
            self._model.named_modules(), pruning_method=prune.L1Unstructured, amount=0.2
        )
        return self

    def __exit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> None:
        pass


class PruningOptimization(Optimization):
    def __init__(self, config: PruningOptimizationConfig):
        self._config = config

    def apply(self, model: nn.Module) -> OptimizationContext:
        return PruningOptimizationContext(model)


class PruningOptimizationFactory(OptimizationFactory):
    @staticmethod
    def _get_config():
        return PruningOptimizationConfig

    @staticmethod
    def _get_object(config: OptimizationConfig):
        return PruningOptimization(
            check_config_entries(config, PruningOptimizationConfig)
        )
