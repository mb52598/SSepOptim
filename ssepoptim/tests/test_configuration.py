import os
import random
import unittest
from typing import Callable, Optional, Union

import torch.nn as nn

from ssepoptim.base.configuration import BaseConfig, ConfigLoader


def get_valid_test_config():
    class TestConfig(BaseConfig):
        a: bool
        b: float
        c: int
        d: str
        e: Optional[int]
        f: Union[int, None]
        g: list[int]
        h: Callable[[float, int], str]

    return TestConfig


def get_invalid_test_config():
    class TestConfig(BaseConfig):
        a: bool
        b: float
        c: int
        d: str
        e: Optional[int]
        f: Union[int, None]
        g: list[int]
        h: type[nn.Module]

    return TestConfig


def get_missing_test_config():
    class MissingTestConfig(BaseConfig):
        a: bool
        b: float
        c: int
        d: str
        e: Optional[int]
        f: Union[int, None]
        g: list[int]
        h: Callable[[float, int], str]

    return MissingTestConfig


def test_function(a: float, b: int) -> str:
    return f"{a} = {b}"


class TestConfiguration(unittest.TestCase):
    def setUp(self) -> None:
        self._test_config_filename = "_test_config.ini"
        test_config = """
        [Test]
        a = true
        b = 5.123
        c = 231
        d = hello hello hello
        f = 12314
        g = 12,34,56,78
        h = ssepoptim.tests.test_configuration.test_function

        [NotTest]
        a = false
        b = 3.123
        c = 123
        """
        with open(self._test_config_filename, "x") as file:
            file.write(test_config)
        self._config_loader = ConfigLoader(self._test_config_filename)

    def tearDown(self) -> None:
        os.remove(self._test_config_filename)

    def test_reading_config(self) -> None:
        config = self._config_loader.get_config(get_valid_test_config())
        self.assertEqual(config["a"], True)
        self.assertEqual(config["b"], 5.123)
        self.assertEqual(config["c"], 231)
        self.assertEqual(config["d"], "hello hello hello")
        self.assertIs(config["e"], None)
        self.assertEqual(config["f"], 12314)
        self.assertEqual(config["g"], [12, 34, 56, 78])
        random_float = random.random()
        random_int = int.from_bytes(random.randbytes(4), "little", signed=True)
        self.assertEqual(
            config["h"](random_float, random_int),
            test_function(random_float, random_int),
        )

    def test_invalid_config(self) -> None:
        self.assertRaises(
            RuntimeError,
            lambda: self._config_loader.get_config(get_invalid_test_config()),
        )

    def test_missing_config(self) -> None:
        self.assertRaises(
            KeyError,
            lambda: self._config_loader.get_config(get_missing_test_config()),
        )
