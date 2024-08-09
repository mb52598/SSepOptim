from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


def chain(value: T, functions: Iterable[Callable[[T], T]]) -> T:
    for function in functions:
        value = function(value)
    return value


def chain_two_values(values: Iterable[T], function: Callable[[T, T], T]) -> T:
    it = iter(values)
    a = next(it)
    try:
        while True:
            b = next(it)
            a = function(a, b)
    except StopIteration:
        pass
    return a
