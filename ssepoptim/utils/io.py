from io import BytesIO, StringIO
from typing import TYPE_CHECKING, Any, BinaryIO, Iterable, TextIO, overload

from ssepoptim.utils.distributed import get_local_rank, is_distributed

if TYPE_CHECKING:
    from _typeshed import (
        FileDescriptorOrPath,
        OpenBinaryMode,
        OpenTextMode,
        ReadableBuffer,
    )
else:
    FileDescriptorOrPath = Any
    OpenBinaryMode = Any
    OpenTextMode = Any
    ReadableBuffer = Any


class DummyBinaryIO(BytesIO):
    def write(self, buffer: ReadableBuffer) -> int:
        return 0

    def writelines(self, lines: Iterable[ReadableBuffer]) -> None:
        pass


class DummyTextIO(StringIO):
    def write(self, s: str) -> int:
        return 0

    def writelines(self, lines: Iterable[str]) -> None:
        pass


@overload
def io_open(file: FileDescriptorOrPath, mode: OpenBinaryMode) -> BinaryIO: ...


@overload
def io_open(file: FileDescriptorOrPath, mode: OpenTextMode) -> TextIO: ...


def _is_write_io_mode(mode: OpenBinaryMode | OpenTextMode) -> bool:
    if "r" in mode and "+" not in mode:
        return False
    return True


def io_open(file: FileDescriptorOrPath, mode: OpenBinaryMode | OpenTextMode = "r"):
    if _is_write_io_mode(mode) and is_distributed() and get_local_rank() != 0:
        context = DummyBinaryIO() if "b" in mode else DummyTextIO()
    else:
        context = open(file, mode)
    return context
