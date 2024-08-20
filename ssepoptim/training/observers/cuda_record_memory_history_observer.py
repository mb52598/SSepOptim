import logging
from typing import Any, cast

import torch

from ssepoptim.training.base import TrainingConfig
from ssepoptim.training.training_observer import TrainingObserver
from ssepoptim.utils.distributed import get_global_rank

logger = logging.getLogger(__name__)


class CudaRecordMemoryHistoryObserver(TrainingObserver):
    def __init__(self, name: str, max_entries: int):
        self._name = name
        self._max_entries = max_entries

    def _is_primary_cuda_device(self, locals: dict[str, Any]):
        train_config = cast(TrainingConfig, locals["train_config"])
        device = cast(torch.device, locals["device"])
        if train_config["distributed_training"] and get_global_rank() != 0:
            return False
        if not device.type.startswith("cuda"):
            logger.warn("Using cuda memory record on non-cuda device")
            return False
        return True

    def on_program_start(self, locals: dict[str, Any]):
        if not self._is_primary_cuda_device(locals):
            return
        torch.cuda.memory._record_memory_history(max_entries=self._max_entries)

    def on_training_epoch_start(self, locals: dict[str, Any]):
        if not self._is_primary_cuda_device(locals):
            return
        epoch: int = locals["epoch"]
        torch.cuda.memory._dump_snapshot(
            f"{self._name}_train_epoch_{epoch}_start.pickle"
        )

    def on_training_epoch_validation(self, locals: dict[str, Any]):
        if not self._is_primary_cuda_device(locals):
            return
        epoch: int = locals["epoch"]
        torch.cuda.memory._dump_snapshot(
            f"{self._name}_train_epoch_{epoch}_valid.pickle"
        )

    def on_training_epoch_end(self, locals: dict[str, Any]):
        if not self._is_primary_cuda_device(locals):
            return
        epoch: int = locals["epoch"]
        torch.cuda.memory._dump_snapshot(f"{self._name}_train_epoch_{epoch}_end.pickle")

    def on_fine_tuning_epoch_start(self, locals: dict[str, Any]):
        if not self._is_primary_cuda_device(locals):
            return
        epoch: int = locals["epoch"]
        torch.cuda.memory._dump_snapshot(
            f"{self._name}_finetune_epoch_{epoch}_start.pickle"
        )

    def on_fine_tuning_epoch_validation(self, locals: dict[str, Any]):
        if not self._is_primary_cuda_device(locals):
            return
        epoch: int = locals["epoch"]
        torch.cuda.memory._dump_snapshot(
            f"{self._name}_finetune_epoch_{epoch}_valid.pickle"
        )

    def on_fine_tuning_epoch_end(self, locals: dict[str, Any]):
        if not self._is_primary_cuda_device(locals):
            return
        epoch: int = locals["epoch"]
        torch.cuda.memory._dump_snapshot(
            f"{self._name}_finetune_epoch_{epoch}_end.pickle"
        )

    def on_testing_end(self, locals: dict[str, Any]):
        if not self._is_primary_cuda_device(locals):
            return
        torch.cuda.memory._dump_snapshot(f"{self._name}_test.pickle")

    def on_program_end(self, locals: dict[str, Any]):
        if not self._is_primary_cuda_device(locals):
            return
        torch.cuda.memory._record_memory_history(enabled=None)
