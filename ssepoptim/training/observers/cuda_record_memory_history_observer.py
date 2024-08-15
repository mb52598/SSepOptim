import logging
from typing import Any, cast

import torch

from ssepoptim.training.base import TrainingConfig
from ssepoptim.training.training_observer import TrainingObserver
from ssepoptim.utils.distributed import get_global_rank

logger = logging.getLogger(__name__)


class CudaRecordMemoryHistoryObserver(TrainingObserver):
    def __init__(self, path: str):
        self._path = path

    def on_program_start(self, locals: dict[str, Any]):
        train_config = cast(TrainingConfig, locals["train_config"])
        device = cast(torch.device, locals["device"])
        if train_config["distributed_training"] and get_global_rank() != 0:
            return
        if device.type.startswith("cuda"):
            torch.cuda.memory._record_memory_history(max_entries=100_000)
        else:
            logger.warn("Using cuda memory record on non-cuda device")

    def on_program_end(self, locals: dict[str, Any]):
        train_config = cast(TrainingConfig, locals["train_config"])
        device = cast(torch.device, locals["device"])
        if train_config["distributed_training"] and get_global_rank() != 0:
            return
        if device.type.startswith("cuda"):
            torch.cuda.memory._dump_snapshot(self._path)
            torch.cuda.memory._record_memory_history(enabled=None)
