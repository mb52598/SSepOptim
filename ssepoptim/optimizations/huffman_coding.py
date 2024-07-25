from typing import Literal

import torch.nn as nn

from ssepoptim.optimization import (
    Optimization,
    OptimizationConfig,
    OptimizationFactory,
)
from ssepoptim.utils.type_checker import check_config_entries


class HuffmanCodingOptimizationConfig(OptimizationConfig):
    pass


class HuffmanCodingOptimization(Optimization):
    def __init__(self, config: HuffmanCodingOptimizationConfig):
        self._config = config

    def apply(self, model: nn.Module) -> nn.Module:
        return model

    @classmethod
    def getType(cls) -> Literal["training", "inference"]:
        return "inference"

    @classmethod
    def requiresFinetune(cls) -> bool:
        return False


class HuffmanCodingOptimizationFactory(OptimizationFactory):
    @staticmethod
    def _get_config():
        return HuffmanCodingOptimizationConfig

    @staticmethod
    def _get_object(config: OptimizationConfig):
        return HuffmanCodingOptimization(
            check_config_entries(config, HuffmanCodingOptimizationConfig)
        )
