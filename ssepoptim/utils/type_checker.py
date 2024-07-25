from typing import Any, Type, TypeVar, cast, get_type_hints

from ssepoptim.base.configuration import BaseConfig

T = TypeVar("T")


def check_object_entries(obj: Any, cls: Type[T]) -> T:
    for k in get_type_hints(cls):
        if not hasattr(obj, k):
            raise RuntimeError(f'Object missing property "{k}"')
    return cast(cls, obj)

def check_config_entries(dictionary: BaseConfig, cls: Type[T]) -> T:
    for k in get_type_hints(cls):
        if k not in dictionary:
            raise RuntimeError(f'Config missing property "{k}"')
    return cast(cls, dictionary)
