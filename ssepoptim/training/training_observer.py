from abc import ABCMeta
from typing import Any, cast, Iterable

from ssepoptim.base.configuration import Constructable


class TrainingObserver(metaclass=ABCMeta):
    def on_program_start(self, locals: dict[str, Any]):
        pass

    def on_program_end(self, locals: dict[str, Any]):
        pass

    def on_training_start(self, locals: dict[str, Any]):
        pass

    def on_training_end(self, locals: dict[str, Any]):
        pass

    def on_training_epoch_start(self, locals: dict[str, Any]):
        pass

    def on_training_epoch_end(self, locals: dict[str, Any]):
        pass

    def on_fine_tuning_start(self, locals: dict[str, Any]):
        pass

    def on_fine_tuning_end(self, locals: dict[str, Any]):
        pass

    def on_fine_tuning_epoch_start(self, locals: dict[str, Any]):
        pass

    def on_fine_tuning_epoch_end(self, locals: dict[str, Any]):
        pass

    def on_testing_start(self, locals: dict[str, Any]):
        pass

    def on_testing_end(self, locals: dict[str, Any]):
        pass


class TrainingObservers(metaclass=ABCMeta):
    def __init__(self, observers: Iterable[Constructable[TrainingObserver]]):
        self._observers = cast(Iterable[TrainingObserver], observers)

    def on_program_start(self, locals: dict[str, Any]):
        for observer in self._observers:
            observer.on_program_start(locals)

    def on_program_end(self, locals: dict[str, Any]):
        for observer in self._observers:
            observer.on_program_end(locals)

    def on_training_start(self, locals: dict[str, Any]):
        for observer in self._observers:
            observer.on_training_start(locals)

    def on_training_end(self, locals: dict[str, Any]):
        for observer in self._observers:
            observer.on_training_end(locals)

    def on_training_epoch_start(self, locals: dict[str, Any]):
        for observer in self._observers:
            observer.on_training_epoch_start(locals)

    def on_training_epoch_end(self, locals: dict[str, Any]):
        for observer in self._observers:
            observer.on_training_epoch_end(locals)

    def on_fine_tuning_start(self, locals: dict[str, Any]):
        for observer in self._observers:
            observer.on_fine_tuning_start(locals)

    def on_fine_tuning_end(self, locals: dict[str, Any]):
        for observer in self._observers:
            observer.on_fine_tuning_end(locals)

    def on_fine_tuning_epoch_start(self, locals: dict[str, Any]):
        for observer in self._observers:
            observer.on_fine_tuning_epoch_start(locals)

    def on_fine_tuning_epoch_end(self, locals: dict[str, Any]):
        for observer in self._observers:
            observer.on_fine_tuning_epoch_end(locals)

    def on_testing_start(self, locals: dict[str, Any]):
        for observer in self._observers:
            observer.on_testing_start(locals)

    def on_testing_end(self, locals: dict[str, Any]):
        for observer in self._observers:
            observer.on_testing_end(locals)
