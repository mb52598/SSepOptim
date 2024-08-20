from typing import Any, TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparisonT
else:
    SupportsRichComparisonT = Any


def min_iter(
    iters: Sequence[Sequence[SupportsRichComparisonT]],
) -> list[SupportsRichComparisonT]:
    if len(iters) == 0:
        return []
    return [min(it[i] for it in iters) for i in range(len(iters[0]))]
