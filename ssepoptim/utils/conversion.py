from types import FunctionType
from typing import Any, Callable, Iterable, Mapping, TypeVar


def dict_any_to_str(d: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for k, v in d.items():
        if type(v) is FunctionType or type(v) is type:
            result[k] = "{}.{}".format(v.__module__, v.__name__)
        else:
            result[k] = str(v)
    return result


K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


def flatten_mappings(dicts: Iterable[Mapping[K, V]]) -> dict[K, V]:
    return {k: v for dct in dicts for k, v in dct.items()}


def flatten(lsts: list[list[T]]) -> list[T]:
    return [v for lst in lsts for v in lst]


def fold(iterable: Iterable[T], func: Callable[[T, T], T], start: T) -> T:
    for value in iterable:
        start = func(start, value)
    return start


def without_index(iterable: Iterable[T], index: int) -> list[T]:
    return [value for i, value in enumerate(iterable) if i != index]
