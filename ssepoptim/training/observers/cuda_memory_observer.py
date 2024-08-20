import logging
from typing import Any, cast

import torch

from ssepoptim.training.training_observer import TrainingObserver

logger = logging.getLogger(__name__)


class CudaMemoryObserver(TrainingObserver):
    def __init__(self):
        pass

    def _get_device(self, locals: dict[str, Any]):
        return cast(torch.device, locals["device"])

    def _reset_peak_memory(self, locals: dict[str, Any]):
        device = self._get_device(locals)
        if device.type.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)

    def _log_peak_memory(self, prefix: str, locals: dict[str, Any]):
        device = self._get_device(locals)
        if device.type.startswith("cuda"):
            peak_memory = torch.cuda.max_memory_allocated(device)
            logger.info("%s|CUDA max memory allocated: %d bytes", prefix, peak_memory)

    def on_training_start(self, locals: dict[str, Any]):
        self._reset_peak_memory(locals)

    def on_training_end(self, locals: dict[str, Any]):
        self._log_peak_memory("Training", locals)

    def on_fine_tuning_start(self, locals: dict[str, Any]):
        self._reset_peak_memory(locals)

    def on_fine_tuning_end(self, locals: dict[str, Any]):
        self._log_peak_memory("Fine-Tune", locals)

    def on_testing_start(self, locals: dict[str, Any]):
        self._reset_peak_memory(locals)

    def on_testing_end(self, locals: dict[str, Any]):
        self._log_peak_memory("Test", locals)
