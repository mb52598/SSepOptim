import torch.nn as nn
import torch.optim as optim

from ssepoptim.model import Model, ModelConfig, ModelFactory
from ssepoptim.utils.type_checker import check_config_entries


class TestModelConfig(ModelConfig):
    number_of_speakers: int


class TestModel(Model):
    def __init__(self, config: TestModelConfig):
        self._config = config

    def get_module(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(1, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(3, self._config["number_of_speakers"], kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        return optim.SGD(model.parameters())

    def get_scheduler(
        self, optimizer: optim.Optimizer
    ) -> optim.lr_scheduler.LRScheduler:
        return optim.lr_scheduler.LinearLR(optimizer)


class TestModelFactory(ModelFactory):
    @staticmethod
    def _get_config():
        return TestModelConfig

    @staticmethod
    def _get_object(config: ModelConfig):
        return TestModel(check_config_entries(config, TestModelConfig))
