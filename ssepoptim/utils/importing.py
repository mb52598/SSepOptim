import logging
import os
from importlib import import_module

logger = logging.getLogger(__name__)


def load_local_modules(file: str, package: str | None):
    package_path, module_file_name = file.rsplit(os.sep, maxsplit=1)
    package_name = package if package is not None else ""
    for file in os.listdir(package_path):
        file_path = os.path.join(package_path, file)
        if file == module_file_name or os.path.isdir(file_path):
            continue
        other_module_name = file.removesuffix(".py")
        if file == other_module_name:
            logger.warning(f'Attempted to import a non-python file: "{file}"')
            continue
        import_module(f"{package_name}.{other_module_name}")
