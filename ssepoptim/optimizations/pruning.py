from typing import Literal

import torch.nn as nn
from torch.ao.pruning.sparsifier.weight_norm_sparsifier import WeightNormSparsifier
import torch.nn.utils.prune as prune

from ssepoptim.optimization import (
    Optimization,
    OptimizationConfig,
    OptimizationFactory,
)
from ssepoptim.utils.type_checker import check_config_entries


class PruningOptimizationConfig(OptimizationConfig):
    pass


class PruningOptimization(Optimization):
    def __init__(self, config: PruningOptimizationConfig):
        self._config = config

    def apply(self, model: nn.Module) -> nn.Module:
        prune.global_unstructured(
            model.named_modules(), pruning_method=prune.L1Unstructured, amount=0.2
        )
        return model

    @classmethod
    def getType(cls) -> Literal["training", "inference"]:
        return "training"

    @classmethod
    def requiresFinetune(cls) -> bool:
        return False


class PruningOptimizationFactory(OptimizationFactory):
    @staticmethod
    def _get_config():
        return PruningOptimizationConfig

    @staticmethod
    def _get_object(config: OptimizationConfig):
        return PruningOptimization(
            check_config_entries(config, PruningOptimizationConfig)
        )
