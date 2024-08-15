from types import FunctionType
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Protocol,
    TypeVar,
)


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
Tcov = TypeVar("Tcov", covariant=True)


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


def convert_not_none(value: K | None, func: Callable[[K], V]) -> V | None:
    if value is not None:
        result = func(value)
    else:
        result = None
    return result


def to_none(_: Any) -> None:
    return None


class IterableLength(Protocol[Tcov]):
    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[Tcov]: ...


def take_percentage(
    iterable: IterableLength[T], percentage: float
) -> Generator[T, None, None]:
    items_to_take = int(len(iterable) * percentage)
    iterator = iter(iterable)
    while True:
        if items_to_take <= 0:
            break
        try:
            item = next(iterator)
        except StopIteration:
            break
        yield item
        items_to_take -= 1
