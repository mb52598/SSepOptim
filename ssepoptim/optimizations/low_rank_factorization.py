from typing import Literal

import torch.nn as nn

from ssepoptim.optimization import (
    Optimization,
    OptimizationConfig,
    OptimizationFactory,
)
from ssepoptim.utils.type_checker import check_config_entries


class LowRankFactorizationOptimizationConfig(OptimizationConfig):
    method: Literal["SVD", "CP", "Tucker"]


class LowRankFactorizationOptimization(Optimization):
    def __init__(self, config: LowRankFactorizationOptimizationConfig):
        self._config = config

    def apply(self, model: nn.Module) -> nn.Module:
        return model

    @classmethod
    def getType(cls) -> Literal["training", "inference"]:
        return "inference"

    @classmethod
    def requiresFinetune(cls) -> bool:
        return False


class LowRankFactorizationOptimizationFactory(OptimizationFactory):
    @staticmethod
    def _get_config():
        return LowRankFactorizationOptimizationConfig

    @staticmethod
    def _get_object(config: OptimizationConfig):
        return LowRankFactorizationOptimization(
            check_config_entries(config, LowRankFactorizationOptimizationConfig)
        )
