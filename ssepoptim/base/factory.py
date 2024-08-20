from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Generic, Self, Type, TypeVar

from ssepoptim.base.configuration import BaseConfig

C = TypeVar("C", bound=BaseConfig)


T = TypeVar("T")


class Factory(Generic[C, T], metaclass=ABCMeta):
    @classmethod
    def _get_subclasses(cls):
        return cls.__subclasses__()

    @classmethod
    def _get_subclass(cls, name: str) -> type[Self]:
        subclasses = [
            subcls for subcls in cls._get_subclasses() if name in subcls.__name__
        ]
        match len(subclasses):
            case 0:
                raise RuntimeError(
                    'Unable to find class "{}" in {}'.format(name, __class__.__name__)
                )
            case 1:
                return subclasses[0]
            case _:
                raise RuntimeError(
                    'Found "{}" classes matching name "{}"'.format(
                        len(subclasses), name
                    )
                )

    @classmethod
    def list_entries(cls) -> list[str]:
        return [
            subclass.__name__.removesuffix("Factory")
            for subclass in cls._get_subclasses()
        ]

    @staticmethod
    @abstractmethod
    def _get_config() -> Type[C]: ...

    @staticmethod
    @abstractmethod
    def _get_object(config: C) -> T: ...

    @classmethod
    def get_config(cls, name: str) -> Type[C]:
        return cls._get_subclass(name)._get_config()

    @classmethod
    def get_object(cls, name: str, config: C) -> T:
        return cls._get_subclass(name)._get_object(config)
