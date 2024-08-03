from typing import Literal

import torch
import torch.ao.quantization as quantization
import torch.nn as nn

from ssepoptim.optimization import (
    Optimization,
    OptimizationConfig,
    OptimizationFactory,
)
from ssepoptim.utils.type_checker import check_config_entries


class QuantizationOptimizationConfig(OptimizationConfig):
    dtype: str


class QuantizationOptimization(Optimization):
    def __init__(self, config: QuantizationOptimizationConfig):
        self._config = config

    def apply(self, model: nn.Module) -> nn.Module:
        quantized_model = quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        return quantized_model

    @classmethod
    def getType(cls) -> Literal["training", "inference"]:
        return "training"

    @classmethod
    def requiresFinetune(cls) -> bool:
        return False


class QuantizationOptimizationFactory(OptimizationFactory):
    @staticmethod
    def _get_config():
        return QuantizationOptimizationConfig

    @staticmethod
    def _get_object(config: OptimizationConfig):
        return QuantizationOptimization(
            check_config_entries(config, QuantizationOptimizationConfig)
        )
