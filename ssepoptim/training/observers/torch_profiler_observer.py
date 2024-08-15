from typing import Any, cast

import torch

from ssepoptim.training.base import TrainingConfig
from ssepoptim.training.training_observer import TrainingObserver
from ssepoptim.utils.distributed import get_global_rank


class TorchProfilerObserver(TrainingObserver):
    def __init__(self, path: str, wait: int, warmup: int, active: int, repeat: int):
        self._path = path
        self._wait = wait
        self._warmup = warmup
        self._active = active
        self._repeat = repeat

    @staticmethod
    def _switch_matplotlib_backend():
        import matplotlib

        matplotlib.use("Agg")

    def on_training_start(self, locals: dict[str, Any]):
        train_config = cast(TrainingConfig, locals["train_config"])
        device = cast(torch.device, locals["device"])
        if train_config["distributed_training"] and get_global_rank() != 0:
            return
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type.startswith("cuda"):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self._wait,
                warmup=self._warmup,
                active=self._active,
                repeat=self._repeat,
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ).__enter__()

    def on_training_epoch_start(self, locals: dict[str, Any]):
        self._profiler.step()

    def on_training_end(self, locals: dict[str, Any]):
        train_config = cast(TrainingConfig, locals["train_config"])
        if train_config["distributed_training"] and get_global_rank() != 0:
            return
        self._profiler.__exit__(None, None, None)
        self._switch_matplotlib_backend()
        self._profiler.export_memory_timeline(self._path)
