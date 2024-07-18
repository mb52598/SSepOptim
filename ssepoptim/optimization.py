from abc import ABCMeta, abstractmethod
from types import TracebackType
from typing import Optional, Protocol, Self, Type

import torch.nn as nn

from ssepoptim.base.configuration import BaseConfig
from ssepoptim.base.factory import Factory


class OptimizationConfig(BaseConfig):
    pass


class OptimizationContext(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> None: ...


class Optimization(metaclass=ABCMeta):
    @abstractmethod
    def apply(self, model: nn.Module) -> OptimizationContext: ...


class OptimizationFactory(Factory[OptimizationConfig, Optimization], metaclass=ABCMeta):
    pass
