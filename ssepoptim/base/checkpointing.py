import os
from datetime import datetime
from typing import Any, Literal, Optional, overload

import torch

from ssepoptim.utils.io import io_open


class Checkpointer:
    TIME_METADATA = "time"
    _JOIN_CHARACTER = "|"
    _NAME_CHARACTER = "="

    def __init__(self, path: str, device: Optional[torch.device] = None):
        self._path = path
        self._device = device
        os.makedirs(self._path, exist_ok=True)

    @staticmethod
    def _create_timestr() -> str:
        return datetime.now().isoformat()

    @classmethod
    def _get_partial_checkpoint_filename(
        cls, module_name: str, visible_metadata: dict[str, str]
    ) -> str:
        excluded_characters = (cls._JOIN_CHARACTER, cls._NAME_CHARACTER)
        # Check for excluded characters in name
        for s in excluded_characters:
            if s in module_name:
                raise ValueError(f'Name cannot contain string "{s}"')
        # Check for time key in visible_metadata
        if cls.TIME_METADATA in visible_metadata:
            raise ValueError(
                f'"{cls.TIME_METADATA}" key is reserved and cannot be used as a metadata entry'
            )
        # Add each (key, value) pair to the file name
        result: list[str] = [module_name]
        for k in sorted(visible_metadata.keys()):
            v = visible_metadata[k]
            for s in excluded_characters:
                if s in k:
                    raise ValueError(
                        f'Invalid metadata key "{k}" as it contains string "{s}"'
                    )
                if s in v:
                    raise ValueError(
                        f'Invalid metadata value "{v}" as it contains string "{s}"'
                    )
            result.append(f"{k}{cls._NAME_CHARACTER}{v}")
        result.append(f"{cls.TIME_METADATA}{cls._NAME_CHARACTER}")
        return cls._JOIN_CHARACTER.join(result)

    @classmethod
    def _create_checkpoint_filename(
        cls, module_name: str, visible_metadata: dict[str, str]
    ) -> str:
        return (
            cls._get_partial_checkpoint_filename(module_name, visible_metadata)
            + cls._create_timestr()
        )

    @classmethod
    def _parse_checkpoint_modulename_and_visible_metadata(
        cls, checkpoint_filename: str
    ) -> tuple[str, dict[str, str]]:
        module_name, *parts = checkpoint_filename.split(cls._JOIN_CHARACTER)
        metadata: dict[str, str] = {}
        for part in parts:
            k, v = part.split(cls._NAME_CHARACTER, maxsplit=1)
            metadata[k] = v
        return module_name, metadata

    def get_checkpoints(self) -> list[str]:
        return os.listdir(self._path)

    def _get_checkpoint_path(self, checkpoint_filename: str) -> str:
        return os.path.join(self._path, checkpoint_filename)

    @overload
    def _load_checkpoint_file(
        self, checkpoint_filename: str, load_data: Literal[True]
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...

    @overload
    def _load_checkpoint_file(
        self, checkpoint_filename: str, load_data: Literal[False]
    ) -> dict[str, Any]: ...

    def _load_checkpoint_file(
        self, checkpoint_filename: str, load_data: bool
    ) -> tuple[dict[str, Any], dict[str, Any]] | dict[str, Any]:
        path = self._get_checkpoint_path(checkpoint_filename)
        with io_open(path, "rb") as file:
            hidden_metadata = torch.load(
                file, map_location=self._device, weights_only=False
            )
            if load_data:
                data = torch.load(file, map_location=self._device, weights_only=False)
                return hidden_metadata, data
            return hidden_metadata

    @overload
    def _load_checkpoint(
        self, checkpoint_filename: str, load_data: Literal[True]
    ) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]: ...

    @overload
    def _load_checkpoint(
        self, checkpoint_filename: str, load_data: Literal[False]
    ) -> tuple[dict[str, str], dict[str, Any]]: ...

    def _load_checkpoint(
        self, checkpoint_filename: str, load_data: bool
    ) -> (
        tuple[dict[str, str], dict[str, Any], dict[str, Any]]
        | tuple[dict[str, str], dict[str, Any]]
    ):
        _, visible_metadata = self._parse_checkpoint_modulename_and_visible_metadata(
            checkpoint_filename
        )
        if load_data:
            hidden_metadata, data = self._load_checkpoint_file(
                checkpoint_filename, load_data=True
            )
            return visible_metadata, hidden_metadata, data
        hidden_metadata = self._load_checkpoint_file(
            checkpoint_filename, load_data=False
        )
        return visible_metadata, hidden_metadata

    def _save_checkpoint(
        self,
        checkpoint_filename: str,
        hidden_metadata: dict[str, Any],
        data: dict[str, Any],
    ):
        path = self._get_checkpoint_path(checkpoint_filename)
        with io_open(path, "xb") as file:
            torch.save(hidden_metadata, file, _use_new_zipfile_serialization=False)
            torch.save(data, file, _use_new_zipfile_serialization=False)

    def _search_checkpoints_metadata(
        self,
        module_name: str,
        visible_metadata: dict[str, str],
        hidden_metadata: dict[str, Any],
        exact_match: bool,
    ) -> list[str]:
        checkpoint_filenames: list[str] = []
        for checkpoint_filename in self.get_checkpoints():
            # Parse module name and visible metadata from checkpoint name
            checkpoint_module_name, checkpoint_visible_metadata = (
                self._parse_checkpoint_modulename_and_visible_metadata(
                    checkpoint_filename
                )
            )
            # If its not our module, skip
            if module_name != checkpoint_module_name:
                continue
            success = True
            # Check if visible_metadata is equal to checkpoint_visible_metadata
            if exact_match:
                success = visible_metadata == checkpoint_visible_metadata
            # Check if visible_metadata is a subset of checkpoint_visible_metadata
            else:
                for k, v in visible_metadata.items():
                    if k == self.TIME_METADATA:
                        continue
                    if (
                        k not in checkpoint_visible_metadata
                        or checkpoint_visible_metadata[k] != v
                    ):
                        success = False
                        break
            # Stop if not successful
            if not success:
                continue
            # Load hidden metadata
            checkpoint_hidden_metadata = self._load_checkpoint_file(
                checkpoint_filename, load_data=False
            )
            # Check if hidden_metadata is equal to checkpoint_hidden_metadata
            if exact_match:
                success = hidden_metadata == checkpoint_hidden_metadata
            else:
                # Check if hidden_metadata is a subset of checkpoint_hidden_metadata
                for k, v in hidden_metadata.items():
                    if (
                        k not in checkpoint_hidden_metadata
                        or checkpoint_hidden_metadata[k] != v
                    ):
                        success = False
                        break
            # If both are true save it
            if success:
                checkpoint_filenames.append(checkpoint_filename)
        return checkpoint_filenames

    def search_checkpoints(
        self,
        module_name: str,
        visible_metadata: dict[str, str] = {},
        hidden_metadata: dict[str, Any] = {},
        desc_sort_by: Optional[str] = None,
    ) -> list[str]:
        result = self._search_checkpoints_metadata(
            module_name, visible_metadata, hidden_metadata, exact_match=False
        )
        if desc_sort_by is not None:
            result = sorted(
                result,
                key=lambda x: self._parse_checkpoint_modulename_and_visible_metadata(x)[
                    1
                ][desc_sort_by],
                reverse=True,
            )
        return result

    def save_checkpoint(
        self,
        module_name: str,
        visible_metadata: dict[str, str],
        hidden_metadata: dict[str, Any],
        data: dict[str, Any],
    ):
        path = self._create_checkpoint_filename(module_name, visible_metadata)
        self._save_checkpoint(path, hidden_metadata, data)

    def get_checkpoint_metadata(
        self, checkpoint_filename: str
    ) -> tuple[dict[str, str], dict[str, Any]]:
        return self._load_checkpoint(checkpoint_filename, load_data=False)

    def load_checkpoint(
        self, checkpoint_filename: str
    ) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return self._load_checkpoint(checkpoint_filename, load_data=True)
