import importlib
import inspect
import re
from collections.abc import Callable
from configparser import ConfigParser, ExtendedInterpolation, SectionProxy
from types import NoneType
from typing import (
    Any,
    Generic,
    Literal,
    Protocol,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from ssepoptim.utils.io import io_open


class BaseConfig(TypedDict):
    pass


class AtleastStrArgumentConstructorClass(Protocol):
    def __init__(self, arg: str, *args: Any):
        pass


T = TypeVar("T")
TC = TypeVar("TC", bound=BaseConfig)


class Constructable(Generic[T]):
    pass


class ConfigLoader:
    _SPECIAL_KEY = "#key"

    def __init__(self, path: str):
        self._parser = ConfigParser(
            interpolation=ExtendedInterpolation(),
        )
        with io_open(path, "r") as file:
            self._parser.read_file(file)

    @staticmethod
    def _split_respecting_brackets(sep: str, string: str, maxsplit: int = 0):
        return re.split(
            sep + r"(?![^{]*})(?![^\[]*\])(?![^\(]*\))", string, maxsplit=maxsplit
        )

    @staticmethod
    def _import_object(path: str):
        module_path, object_name = path.rsplit(".", maxsplit=1)
        module = importlib.import_module(module_path)
        return getattr(module, object_name)

    @staticmethod
    def _check_key(key: str, value: str | None) -> str:
        if value is None:
            raise RuntimeError(f'Declared parameter "{key}" not found in configuration')
        return value

    @classmethod
    def _cast_key_type(
        cls, model: SectionProxy, key: str, value: str | None, keytype: Type[T]
    ) -> T:
        value = cls._check_key(key, value)
        model[cls._SPECIAL_KEY] = value
        if keytype is bool:
            result_value = model.getboolean(cls._SPECIAL_KEY)
        elif keytype is float:
            result_value = model.getfloat(cls._SPECIAL_KEY)
        elif keytype is int:
            result_value = model.getint(cls._SPECIAL_KEY)
        else:
            result_value = cast(Type[AtleastStrArgumentConstructorClass], keytype)(
                model.get(cls._SPECIAL_KEY)
            )
        result_type = cast(type[Any], type(result_value))
        if result_type is not keytype:
            raise RuntimeError(
                'Unable to convert "{}" to type "{}", got "{}"'.format(
                    key, keytype.__name__, result_type.__name__
                )
            )
        return cast(T, result_value)

    @classmethod
    def _parse_origin(
        cls, model: SectionProxy, key: str, value: str | None, keytype: Type[T]
    ) -> Any:
        origin = get_origin(keytype)
        if origin is None:
            return cls._cast_key_type(model, key, value, keytype)
        elif origin is Union:
            args = get_args(keytype)
            if len(args) != 2 or NoneType not in args:
                raise RuntimeError(
                    'Unsupported union with types "{}"'.format(
                        ", ".join(arg.__name__ for arg in args)
                    )
                )
            arg_type = args[0] if args[1] is NoneType else args[1]
            if value is None:
                return None
            else:
                return cls._parse_origin(model, key, value, arg_type)
        elif origin is type:
            value = cls._check_key(key, value)
            imported_class = cls._import_object(value)
            if not inspect.isclass(imported_class):
                raise RuntimeError(
                    'Imported object "{}" is not a class'.format(
                        imported_class.__name__
                    )
                )
            defined_class = get_args(keytype)[0]
            if not issubclass(imported_class, defined_class):
                raise RuntimeError(
                    'Imported class "{}" doesn\'t match the specified type "{}"'.format(
                        imported_class.__name__, defined_class.__name__
                    )
                )
            return imported_class
        elif origin is Constructable:
            value = cls._check_key(key, value)
            m = re.match(r"^(?P<classname>.*?)\((?P<arguments>.*?)\)$", value)
            if m is None:
                raise RuntimeError(
                    f'Invalid Constructible object pattern for key "{key}"'
                )
            d = m.groupdict()
            defined_class = get_args(keytype)[0]
            value_class = cls._parse_origin(
                model, key, d["classname"], type[defined_class]
            )
            parameter_values_string = cls._split_respecting_brackets(
                ",", d["arguments"]
            )
            if len(parameter_values_string) == 1 and not parameter_values_string[0]:
                parameter_values_string = []
            signature = inspect.signature(value_class.__init__)
            parameter_annotataions = [
                p.annotation for p in signature.parameters.values() if p.name != "self"
            ]
            if len(parameter_values_string) != len(parameter_annotataions):
                raise RuntimeError(
                    'Number of provided({}) and number of required({}) parameters mismatch for key "{}"'.format(
                        len(parameter_values_string), len(parameter_annotataions), key
                    )
                )
            parameters = [
                cls._parse_origin(model, key, parameter_string, parameter_annotation)
                for parameter_string, parameter_annotation in zip(
                    parameter_values_string, parameter_annotataions
                )
            ]
            return value_class(*parameters)
        elif origin is tuple:
            value = cls._check_key(key, value)
            value = value.lstrip("[").rstrip("]")
            values_string = cls._split_respecting_brackets(",", value)
            defined_types = get_args(keytype)
            if len(values_string) != len(defined_types):
                raise RuntimeError(
                    'Number of provided({}) and number of required({}) arguments mismatch for key "{}"'.format(
                        len(values_string), len(defined_types), key
                    )
                )
            return tuple(
                cls._parse_origin(model, key, value_string, defined_type)
                for value_string, defined_type in zip(values_string, defined_types)
            )
        elif origin is list:
            defined_type = get_args(keytype)[0]
            result_value: list[Type[Any]] = []
            if value is not None:
                value = value.lstrip("[").rstrip("]")
                for element in cls._split_respecting_brackets(",", value):
                    result_value.append(
                        cls._parse_origin(model, key, element, defined_type)
                    )
            return result_value
        elif origin is Callable:
            value = cls._check_key(key, value)
            imported_function = cls._import_object(value)
            arg_types, return_type = cast(
                tuple[list[type[Any]], type[Any]], get_args(keytype)
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
            return imported_function
        elif origin is Literal:
            args = get_args(keytype)
            if len(args) == 0:
                raise RuntimeError(
                    'Invalid literal type "{}" for key "{}"'.format(
                        keytype.__name__, key
                    )
                )
            arg_type = cast(type[Any], type(args[0]))
            result_value = cls._cast_key_type(model, key, value, arg_type)
            if result_value not in args:
                raise RuntimeError(
                    'Provided value "{}" not in Literal arguments for key "{}", possible values are: {}'.format(
                        result_value, key, args
                    )
                )
            return result_value
        else:
            raise RuntimeError('Unsupported origin "{}"'.format(origin.__name__))

    def get_config(self, cls: Type[TC]) -> TC:
        model = self._parser[cls.__name__.removesuffix("Config")]
        result = cls()
        for key, keytype in get_type_hints(cls).items():
            value = model.get(key)
            result[key] = self._parse_origin(model, key, value, keytype)
        return result
