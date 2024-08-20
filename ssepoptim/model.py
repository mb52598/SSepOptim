from abc import ABCMeta, abstractmethod

import torch.nn as nn
import torch.optim as optim

from ssepoptim.base.configuration import BaseConfig
from ssepoptim.base.factory import Factory


class ModelConfig(BaseConfig):
    pass


class Model(metaclass=ABCMeta):
    @abstractmethod
    def get_module(self) -> nn.Module: ...

    @abstractmethod
    def get_optimizer(self, model: nn.Module) -> optim.Optimizer: ...

    @abstractmethod
    def get_scheduler(
        self, optimizer: optim.Optimizer
    ) -> optim.lr_scheduler.LRScheduler: ...


class ModelFactory(Factory[ModelConfig, Model], metaclass=ABCMeta):
    pass
