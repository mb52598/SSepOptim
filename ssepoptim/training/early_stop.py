from abc import ABCMeta, abstractmethod
from typing import Any


class EarlyStop(metaclass=ABCMeta):
    @abstractmethod
    def should_early_stop(self, locals: dict[str, Any]) -> bool: ...


class DummyEarlyStop(EarlyStop):
    def __init__(self):
        pass

    def should_early_stop(self, locals: dict[str, Any]):
        return False


class UnchangingValidationEarlyStop(EarlyStop):
    def __init__(self, patience: int):
        self._patience = patience
        self._counter = 0
        self._min_valid_avg_loss = float("inf")

    def should_early_stop(self, locals: dict[str, Any]) -> bool:
        valid_avg_loss: float = locals["valid_avg_loss"]
        if valid_avg_loss < self._min_valid_avg_loss:
            self._min_valid_avg_loss = valid_avg_loss
            self._counter = 0
        elif valid_avg_loss > self._min_valid_avg_loss:
            self._counter += 1
            if self._counter > self._patience:
                return True
        return False
