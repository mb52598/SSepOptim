import os
from datetime import datetime
from typing import Any, Optional

import torch

from ssepoptim.base.configuration import BaseConfig


class CheckpointerConfig(BaseConfig):
    path: str


class Checkpointer:
    TIME_METADATA = "time"
    _JOIN_CHARACTER = "|"
    _NAME_CHARACTER = "="

    def __init__(
        self, config: CheckpointerConfig, folders_metadata: dict[str, str] = {}
    ):
        self._path = os.path.join(
            config["path"],
            *[f"{k}{self._NAME_CHARACTER}{v}" for k, v in folders_metadata.items()],
        )
        os.makedirs(self._path, exist_ok=True)

    def get_checkpoints(self) -> list[str]:
        return os.listdir(self._path)

    def _get_checkpoint_path(self, checkpoint_filename: str) -> str:
        return os.path.join(self._path, checkpoint_filename)

    def _load_checkpoint(self, checkpoint_filename: str) -> dict[str, Any]:
        path = self._get_checkpoint_path(checkpoint_filename)
        with open(path, "rb") as file:
            return torch.load(file)

    def _save_checkpoint(
        self,
        checkpoint_filename: str,
        minor_metadata: dict[str, Any],
    ):
        path = self._get_checkpoint_path(checkpoint_filename)
        with open(path, "xb") as file:
            torch.save(minor_metadata, file)

    @staticmethod
    def _create_timestr() -> str:
        return datetime.now().isoformat()

    @classmethod
    def _get_partial_checkpoint_filename(
        cls, module_name: str, major_metadata: dict[str, str]
    ) -> str:
        excluded_characters = (cls._JOIN_CHARACTER, cls._NAME_CHARACTER)
        # Check for excluded characters in name
        for s in excluded_characters:
            if s in module_name:
                raise ValueError(f'Name cannot contain string "{s}"')
        # Check for time key in major_metadata
        if cls.TIME_METADATA in major_metadata:
            raise ValueError(
                f'"{cls.TIME_METADATA}" key is reserved and cannot be used as a metadata entry'
            )
        result: list[str] = [module_name]
        for k in sorted(major_metadata.keys()):
            v = major_metadata[k]
            for s in excluded_characters:
                if s in k or s in v:
                    raise ValueError(
                        f'Invalid metadata value "{k}" or "{v}" as they contain string "{s}"'
                    )
            result.append(f"{k}{cls._NAME_CHARACTER}{v}")
        result.append(f"{cls.TIME_METADATA}{cls._NAME_CHARACTER}")
        return cls._JOIN_CHARACTER.join(result)

    @classmethod
    def _get_checkpoint_modulename_and_majormetadata(
        cls, checkpoint_filename: str
    ) -> tuple[str, dict[str, str]]:
        module_name, *parts = checkpoint_filename.split(cls._JOIN_CHARACTER)
        metadata: dict[str, str] = {}
        for part in parts:
            k, v = part.split(cls._NAME_CHARACTER, maxsplit=1)
            metadata[k] = v
        return module_name, metadata

    def _exact_search_checkpoints(
        self, module_name: str, major_metadata: dict[str, str]
    ) -> list[str]:
        checkpoint_partial_filename = self._get_partial_checkpoint_filename(
            module_name, major_metadata
        )
        return [
            file
            for file in self.get_checkpoints()
            if file.startswith(checkpoint_partial_filename)
        ]

    def _approximate_search_checkpoints(
        self,
        module_name: str,
        major_metadata: dict[str, str],
    ) -> list[str]:
        files: list[str] = []
        for file in self.get_checkpoints():
            checkpoint_module_name, checkpoint_major_metadata = (
                self._get_checkpoint_modulename_and_majormetadata(file)
            )
            if module_name != checkpoint_module_name:
                continue
            success = True
            for k, v in checkpoint_major_metadata.items():
                if k == self.TIME_METADATA:
                    continue
                if k not in major_metadata or major_metadata[k] != v:
                    success = False
                    break
            if success:
                files.append(file)
        return files

    def _create_checkpoint_filename(
        self, module_name: str, major_metadata: dict[str, str]
    ) -> str:
        return (
            self._get_partial_checkpoint_filename(module_name, major_metadata)
            + self._create_timestr()
        )

    def get_minor_metadata(self, checkpoint_filename: str) -> dict[str, Any]:
        return self._load_checkpoint(checkpoint_filename)

    def search_minor_metadata(
        self,
        module_name: str,
        major_metadata: dict[str, str] = {},
        pick_latest: bool = False,
    ) -> dict[str, Any]:
        checkpoint_filenames = self._exact_search_checkpoints(
            module_name, major_metadata
        )
        if len(checkpoint_filenames) == 1:
            return self.get_minor_metadata(checkpoint_filenames[0])
        elif len(checkpoint_filenames) > 0 and pick_latest:
            return self.get_minor_metadata(
                sorted(checkpoint_filenames, reverse=True)[0]
            )
        checkpoint_filenames = self._approximate_search_checkpoints(
            module_name, major_metadata
        )
        if len(checkpoint_filenames) == 1:
            return self.get_minor_metadata(checkpoint_filenames[0])
        raise ValueError(
            f"Unable to load class {module_name} found {len(checkpoint_filenames)} modules"
        )

    def save(
        self,
        module_name: str,
        major_metadata: dict[str, str],
        minor_metadata: dict[str, Any],
    ):
        path = self._create_checkpoint_filename(module_name, major_metadata)
        self._save_checkpoint(path, minor_metadata)

    def search_checkpoints(
        self,
        module_name: str,
        major_metadata: dict[str, str] = {},
        desc_sort_by: Optional[str] = None,
    ) -> list[str]:
        result = self._approximate_search_checkpoints(module_name, major_metadata)
        if desc_sort_by is not None:
            result = sorted(
                result,
                key=lambda x: self.get_major_metadata(x)[desc_sort_by],
                reverse=True,
            )
        return result

    def get_major_metadata(self, checkpoint_filename: str) -> dict[str, str]:
        return self._get_checkpoint_modulename_and_majormetadata(checkpoint_filename)[1]
