from typing import Any, cast

import torch

from ssepoptim.training.training_observer import TrainingObserver


class TorchProfilerObserver(TrainingObserver):
    def __init__(self, path: str):
        self._path = path

    def on_training_start(self, locals: dict[str, Any]):
        device = cast(torch.device, locals["device"])
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type.startswith("cuda"):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ).__enter__()

    def on_training_epoch_start(self, locals: dict[str, Any]):
        self._profiler.step()

    def on_training_end(self, locals: dict[str, Any]):
        self._profiler.__exit__(None, None, None)
        import matplotlib
        matplotlib.use('Agg')
        self._profiler.export_memory_timeline(self._path)
