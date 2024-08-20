import logging
from typing import Any, cast

import torch
import torch.nn as nn

from ssepoptim.dataset import SpeechSeparationDataset
from ssepoptim.training.base import TrainingConfig, get_module_latency_and_throughput
from ssepoptim.training.training_observer import TrainingObserver
from ssepoptim.utils.module_stats import (
    get_module_parameters_count,
    get_module_parameters_memory_usage,
)

logger = logging.getLogger(__name__)


class ModuleStatsObserver(TrainingObserver):
    def __init__(self):
        self._latency_points = 1
        self._latency_calculation_repetitions = 1

    def _calculate_latency_and_throughput(
        self, module: nn.Module, locals: dict[str, Any]
    ):
        # Steal the program configuration
        dataset = cast(SpeechSeparationDataset, locals["dataset"]).get_train()
        device = cast(torch.device, locals["device"])
        train_config = cast(TrainingConfig, locals["train_config"])
        # Calculate latency
        return get_module_latency_and_throughput(
            module, dataset, train_config["batch_size"], device
        )

    def _log_module_stats(self, prefix: str, locals: dict[str, Any]):
        # Steal the module
        module = cast(nn.Module, locals["module"])
        param_count = get_module_parameters_count(module)
        memory_usage = get_module_parameters_memory_usage(module)
        latency, throughput = self._calculate_latency_and_throughput(module, locals)
        logger.info(
            "%s|Paramer count: %d|Memory usage: %d bytes|Latency: %f ns|Throughput: %f batch/s",
            prefix,
            param_count,
            memory_usage,
            latency,
            throughput,
        )

    def on_training_end(self, locals: dict[str, Any]):
        self._log_module_stats("Training", locals)

    def on_fine_tuning_end(self, locals: dict[str, Any]):
        self._log_module_stats("Fine-Tune", locals)

    def on_testing_end(self, locals: dict[str, Any]):
        self._log_module_stats("Test", locals)
