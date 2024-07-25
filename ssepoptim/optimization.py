from abc import ABCMeta, abstractmethod
from typing import Iterable, Literal

import torch.nn as nn

from ssepoptim.base.configuration import BaseConfig
from ssepoptim.base.factory import Factory
from ssepoptim.utils.chaining import chain


class OptimizationConfig(BaseConfig):
    pass


class Optimization(metaclass=ABCMeta):
    @abstractmethod
    def apply(self, model: nn.Module) -> nn.Module: ...

    @classmethod
    @abstractmethod
    def getType(cls) -> Literal["training", "inference"]: ...

    @classmethod
    @abstractmethod
    def requiresFinetune(cls) -> bool: ...


class Optimizations(metaclass=ABCMeta):
    @staticmethod
    def apply(
        model: nn.Module,
        optimizations: Iterable[Optimization],
        stage: Literal["train", "test"],
    ) -> nn.Module:
        match stage:
            case "train":
                required_type = "training"
            case "test":
                required_type = "inference"
        return chain(
            model,
            [
                optimization.apply
                for optimization in optimizations
                if optimization.getType() == required_type
            ],
        )

    @staticmethod
    def requireFinetune(optimizations: Iterable[Optimization]) -> bool:
        return any(optimization.requiresFinetune() for optimization in optimizations)


class OptimizationFactory(Factory[OptimizationConfig, Optimization], metaclass=ABCMeta):
    pass
