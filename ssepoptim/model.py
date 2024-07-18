from abc import ABCMeta

import torch.nn as nn

from ssepoptim.base.configuration import BaseConfig
from ssepoptim.base.factory import Factory


class ModelConfig(BaseConfig):
    pass


class ModelFactory(Factory[ModelConfig, nn.Module], metaclass=ABCMeta):
    pass
