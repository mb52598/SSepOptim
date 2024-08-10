from abc import ABCMeta, abstractmethod
from typing import Any, Iterable, Type


class TrainingObserver(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self): ...

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
    def __init__(self, observers: Iterable[Type[TrainingObserver]]):
        self._observers = [observer() for observer in observers]

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
