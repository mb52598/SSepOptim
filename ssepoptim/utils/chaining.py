from typing import Callable, Iterable, TypeVar

T = TypeVar("T")

def chain(value: T, functions: Iterable[Callable[[T], T]]) -> T:
    for function in functions:
        value = function(value)
    return value
