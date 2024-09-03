import os
from typing import Any, Callable, Generator, TypeVar, cast

T = TypeVar("T")


def _parse_scheme_recursion(
    scheme: dict[str, list[T]], index: int, current_values: dict[str, T]
) -> Generator[dict[str, T], None, None]:
    if index == len(scheme):
        yield current_values.copy()
        return
    name = list(scheme.keys())[index]
    values = scheme[name]
    for value in values:
        current_values[name] = value
        yield from _parse_scheme_recursion(scheme, index + 1, current_values)


def _merge_dict(dct: dict[str, T | dict[str, T]]) -> dict[str, T]:
    result: dict[str, T] = {}
    for k, v in dct.items():
        if isinstance(v, dict):
            result.update(cast(dict[str, T], v))
        else:
            result[k] = v
    return result


def _write_file_from_template(
    template_path: str, output_path: str, values: dict[str, str]
):
    with open(template_path, "r") as in_file:
        with open(output_path, "x") as out_file:
            line = " "
            while line:
                line = in_file.readline()
                out_file.write(line)
                if line.startswith("[DEFAULT]"):
                    for name, value in values.items():
                        out_file.write(f"{name} = {value}\n")


def build_folders(
    scheme: dict[str, list[str] | list[dict[str, str]]],
    folders_func: Callable[[str, dict[str, str]], str],
    filename_func: Callable[[str, dict[str, str]], str],
):
    schema = cast(dict[str, list[Any]], scheme)
    for root, _, filenames in os.walk("."):
        for values in _parse_scheme_recursion(schema, 0, {}):
            values = _merge_dict(values)
            for filename in filenames:
                if not filename.endswith("template"):
                    continue
                out_root = folders_func(root, values)
                os.makedirs(out_root, exist_ok=True)
                new_filename = filename_func(filename, values)
                _write_file_from_template(
                    os.path.join(root, filename),
                    os.path.join(out_root, new_filename),
                    values,
                )
