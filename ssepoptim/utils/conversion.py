from types import FunctionType
from typing import Any


def dict_any_to_str(d: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for k, v in d.items():
        if type(v) is FunctionType or type(v) is type:
            result[k] = "{}.{}".format(v.__module__, v.__name__)
        else:
            result[k] = str(v)
    return result
