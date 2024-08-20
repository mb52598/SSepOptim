import time
from types import TracebackType
from typing import Optional, Self, Type


class CtxTimer:
    def __init__(self) -> None:
        self.reset()

    def __enter__(self) -> Self:
        self.reset()
        return self

    def __exit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> None:
        pass

    def reset(self) -> None:
        self.start_time = time.time()

    @property
    def total(self):
        return time.time() - self.start_time
