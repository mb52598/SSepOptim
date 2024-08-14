from abc import ABCMeta, abstractmethod
from typing import Any, Iterable, Literal

import torch.nn as nn

from ssepoptim.base.configuration import BaseConfig
from ssepoptim.base.factory import Factory


class OptimizationConfig(BaseConfig):
    pass


OptimizationStage = Literal[
    "TRAIN_START", "TRAIN_END", "FINETUNE_START", "FINETUNE_END", "TEST_START"
]


class Optimization(metaclass=ABCMeta):
    @abstractmethod
    def apply(
        self, module: nn.Module, stage: OptimizationStage, locals: dict[str, Any]
    ) -> nn.Module: ...

    @abstractmethod
    def requiresFinetune(self) -> bool: ...


class Optimizations(metaclass=ABCMeta):
    @staticmethod
    def apply(
        module: nn.Module,
        optimizations: Iterable[Optimization],
        stage: OptimizationStage,
        locals: dict[str, Any],
    ) -> nn.Module:
        for optimization in optimizations:
            module = optimization.apply(module, stage, locals)
        return module

    @staticmethod
    def requireFinetune(optimizations: Iterable[Optimization]) -> bool:
        return any(optimization.requiresFinetune() for optimization in optimizations)


class OptimizationFactory(Factory[OptimizationConfig, Optimization], metaclass=ABCMeta):
    pass
