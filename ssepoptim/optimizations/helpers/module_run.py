from typing import Any, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ssepoptim.dataset import SpeechSeparationDatasetType
from ssepoptim.training.base import TrainingConfig, collate_fn, sample_dataset
from ssepoptim.utils.context_timer import CtxTimer
from ssepoptim.utils.conversion import take_percentage


def get_local_data(locals: dict[str, Any]):
    dataset = cast(SpeechSeparationDatasetType, locals["train_dataset"])
    device = cast(torch.device, locals["device"])
    train_config = cast(TrainingConfig, locals["train_config"])
    return dataset, device, train_config


def get_dataset_data(locals: dict[str, Any]):
    dataset, device, train_config = get_local_data(locals)
    return (
        sample_dataset(dataset, batch_size=train_config["batch_size"])[0].to(device),
    )


def calibrate_module(module: nn.Module, locals: dict[str, Any]):
    dataset, device, train_config = get_local_data(locals)
    dataloader = DataLoader(
        dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=train_config["num_workers"],
        collate_fn=collate_fn,
        generator=torch.Generator(device),
    )
    timer = CtxTimer()
    with torch.no_grad():
        mix: torch.Tensor
        for mix, _ in dataloader:
            module(mix.to(device))
    return timer.total


def calculate_module_gradients(
    module: nn.Module, locals: dict[str, Any], dataset_percentage: float
):
    dataset, device, train_config = get_local_data(locals)
    loss = train_config["loss"]
    dataloader = DataLoader(
        dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=train_config["num_workers"],
        collate_fn=collate_fn,
        generator=torch.Generator(device),
    )
    module.zero_grad(set_to_none=True)
    timer = CtxTimer()
    mix: torch.Tensor
    target: torch.Tensor
    for mix, target in take_percentage(dataloader, dataset_percentage):
        mix = mix.to(device)
        target = target.to(device)
        separation = module(mix)
        separation_loss = loss(separation, target)
        separation_loss.backward()
    return timer.total
