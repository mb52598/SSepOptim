from typing import Any

import torch.nn as nn

from ssepoptim.optimization import (
    Optimization,
    OptimizationConfig,
    OptimizationFactory,
    OptimizationStage,
)
from ssepoptim.utils.type_checker import check_config_entries


class HuffmanCodingOptimizationConfig(OptimizationConfig):
    pass


class HuffmanCodingOptimization(Optimization):
    def __init__(self, config: HuffmanCodingOptimizationConfig):
        self._config = config

    def apply(
        self, module: nn.Module, stage: OptimizationStage, locals: dict[str, Any]
    ) -> nn.Module:
        return module

    def requiresFinetune(self) -> bool:
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
