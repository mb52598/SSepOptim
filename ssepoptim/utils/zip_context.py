from types import TracebackType
from typing import Iterable, Optional, Protocol, Self, Type


class ContextClass(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> None: ...


class ZipContextClass:
    def __init__(self, contexts: Iterable[ContextClass]) -> None:
        self._contexts = contexts

    def __enter__(self) -> Self:
        for context in self._contexts:
            context.__enter__()
        return self

    def __exit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> None:
        for context in self._contexts:
            context.__exit__(exctype, excinst, exctb)


def zip_context(contexts: Iterable[ContextClass]):
    return ZipContextClass(contexts)
