import importlib
from collections.abc import Callable
from configparser import ConfigParser, ExtendedInterpolation, SectionProxy
from types import NoneType
from typing import (
    Any,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)


class BaseConfig(TypedDict):
    pass


T = TypeVar("T")
TC = TypeVar("TC", bound=BaseConfig)


class ConfigLoader:
    def __init__(self, path: str):
        self._parser = ConfigParser(
            interpolation=ExtendedInterpolation(),
        )
        with open(path, "r") as file:
            self._parser.read_file(file)

    @staticmethod
    def _import_object(path: str):
        module_path, object_name = path.rsplit(".", maxsplit=1)
        module = importlib.import_module(module_path)
        return getattr(module, object_name)

    @staticmethod
    def _parse_key_value_annotation(model: SectionProxy, key: str, value: Type[T]) -> T:
        if key not in model:
            raise RuntimeError(f'Declared parameter "{key}" not found in configuration')
        if value is bool:
            conversion_func_name = "getboolean"
        elif value is float:
            conversion_func_name = "getfloat"
        elif value is int:
            conversion_func_name = "getint"
        elif value is str:
            conversion_func_name = "get"
        else:
            raise RuntimeError('Unsupported type "{}"'.format(value.__name__))
        result_value = getattr(model, conversion_func_name)(key)
        result_type = cast(type[Any], type(result_value))
        if result_type is not value:
            raise RuntimeError(
                'Unable to convert "{}" to type "{}", got "{}"'.format(
                    key, value.__name__, result_type.__name__
                )
            )
        return result_value

    def get_config(self, cls: Type[TC]) -> TC:
        model = self._parser[cls.__name__.removesuffix("Config")]
        result = cls()
        for key, value in get_type_hints(cls).items():
            origin = get_origin(value)
            if origin is None:
                result[key] = self._parse_key_value_annotation(model, key, value)
            elif origin is Union:
                args = value = get_args(value)
                if len(args) != 2 or NoneType not in args:
                    raise RuntimeError(
                        'Unsupported union with types "{}"'.format(
                            ", ".join(arg.__name__ for arg in args)
                        )
                    )
                if key not in model:
                    result[key] = None
                else:
                    result[key] = self._parse_key_value_annotation(model, key, args[0])
            elif origin is type:
                imported_class = self._import_object(
                    self._parse_key_value_annotation(model, key, str)
                )
                defined_class = get_args(value)[0]
                if not issubclass(imported_class, defined_class):
                    raise RuntimeError(
                        'Imported class "{}" doesn\'t match the specified type "{}"'.format(
                            imported_class.__name__, defined_class.__name__
                        )
                    )
                result[key] = imported_class
            elif origin is list:
                defined_type = get_args(value)[0]
                result_value: list[Any] = []
                if key in model:
                    for element in self._parse_key_value_annotation(
                        model, key, str
                    ).split(","):
                        result_value.append(defined_type(element))
                result[key] = result_value
            elif origin is Callable:
                imported_function = self._import_object(
                    self._parse_key_value_annotation(model, key, str)
                )
                arg_types, return_type = cast(
                    tuple[list[type[Any]], type[Any]], get_args(value)
                )
                *defined_arg_types, defined_return_type = get_type_hints(
                    imported_function
                ).values()
                for i, (arg_type, defined_arg_type) in enumerate(
                    zip(arg_types, defined_arg_types), start=1
                ):
                    if arg_type is not defined_arg_type:
                        raise RuntimeError(
                            'Imported function {}. argument type "{}" doesn\'t match specified type "{}"'.format(
                                i, arg_type.__name__, defined_arg_type.__name__
                            )
                        )
                if return_type is not defined_return_type:
                    raise RuntimeError(
                        'Imported function return type "{}" doesn\'t match specified type "{}"'.format(
                            return_type.__name__, defined_return_type.__name__
                        )
                    )
                result[key] = imported_function
            else:
                raise RuntimeError('Unsupported origin "{}"'.format(origin.__name__))
        return result
