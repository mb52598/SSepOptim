import os
import pickle
from typing import Callable, TypeVar

from ssepoptim.utils.io import io_open

T = TypeVar("T")


def load_from_cache(path: str, load_func: Callable[[], T]) -> T:
    if os.path.exists(path):
        with io_open(path, "rb") as file:
            result = pickle.load(file)
    else:
        result = load_func()
        with io_open(path, "xb") as file:
            pickle.dump(result, file)
    return result
