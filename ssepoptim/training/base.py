import random
import time
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import ssepoptim.loss as loss
from ssepoptim.base.checkpointing import Checkpointer
from ssepoptim.base.configuration import BaseConfig, Constructable
from ssepoptim.dataset import SpeechSeparationDatasetConfig, SpeechSeparationDatasetType
from ssepoptim.metrics.base import Metric
from ssepoptim.model import ModelConfig
from ssepoptim.optimization import OptimizationConfig
from ssepoptim.training.training_observer import TrainingObserver
from ssepoptim.training.early_stop import EarlyStop
from ssepoptim.utils.context_timer import CtxTimer
from ssepoptim.utils.conversion import dict_any_to_str
from ssepoptim.utils.torch_utils import synchronize_device

DatasetData = tuple[torch.Tensor, torch.Tensor]
DatasetDataLoader = DataLoader[DatasetData]


# Useful for preventing typing errors
class CheckpointerKeys:
    ID = "id"
    SEED = "seed"
    EPOCH = "epoch"
    TRAIN_AVERAGE_LOSS = "train_avg_loss"
    VALID_AVERAGE_LOSS = "valid_avg_loss"
    MODEL_CONFIG = "model_config"
    OPTIMIZATION_CONFIGS = "optimization_configs"
    DATASET_CONFIG = "dataset_config"
    TRAINING_CONFIG = "training_config"
    MODULE_STATE_DICT = "module_state_dict"
    OPTIMIZER_STATE_DICT = "optimizer_state_dict"
    SCHEDULER_STATE_DICT = "scheduler_state_dict"


class ReducedTrainingConfig(BaseConfig):
    epochs: int
    finetune_epochs: int
    batch_size: int
    clip_grad_norm: Optional[float]
    shuffle: bool
    num_workers: int
    checkpoint_epoch_log: int
    loss: loss.Loss
    convert_loss_to_permutation_invariant: Literal["yes", "greedy", "no"]
    load_last_checkpoint: bool
    save_finetune_checkpoint: bool
    apply_performance_optimizers: bool
    test_only: bool
    test_metrics: list[Metric]
    calculate_test_metrics_improvement: bool
    checkpoints_path: str
    observers: list[Constructable[TrainingObserver]]
    early_stop: Optional[Constructable[EarlyStop]]
    distributed_training: bool
    distributed_find_unused_params: bool


class TrainingConfig(ReducedTrainingConfig):
    id: Optional[str]
    device: Optional[str]
    seed: Optional[int]


def train_loop(
    train_dataloader: DatasetDataLoader,
    module: nn.Module,
    loss: loss.Loss,
    optimizer: optim.Optimizer,
    device: Optional[torch.device],
    clip_grad_norm: Optional[float],
):
    module.train()
    train_loss_sum = torch.zeros(1, device=device)
    timer = CtxTimer()
    mix: torch.Tensor
    target: torch.Tensor
    for mix, target in train_dataloader:
        mix = mix.to(device)
        target = target.to(device)
        separation = module(mix)
        separation_loss = torch.mean(loss(separation, target))
        train_loss_sum += separation_loss.detach()
        optimizer.zero_grad(set_to_none=True)
        separation_loss.backward()
        if clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(module.parameters(), clip_grad_norm)
        optimizer.step()
    train_avg_loss = train_loss_sum.item() / len(train_dataloader)
    return train_avg_loss, timer.total


def valid_loop(
    valid_dataloader: DatasetDataLoader,
    module: nn.Module,
    loss: loss.Loss,
    device: Optional[torch.device],
):
    module.eval()
    valid_loss_sum = torch.zeros(1, device=device)
    timer = CtxTimer()
    with torch.no_grad():
        mix: torch.Tensor
        target: torch.Tensor
        for mix, target in valid_dataloader:
            mix = mix.to(device)
            target = target.to(device)
            separation = module(mix)
            separation_loss = torch.mean(loss(separation, target))
            valid_loss_sum += separation_loss
    valid_avg_loss = valid_loss_sum.item() / len(valid_dataloader)
    return valid_avg_loss, timer.total


def _calculate_metric_improvement(
    metrics: list[Metric],
    mix: torch.Tensor,
    target: torch.Tensor,
    separation: torch.Tensor,
):
    return [
        torch.mean(
            metric(separation, target)
            - metric(mix.expand(-1, target.shape[1], -1), target)
        )
        for metric in metrics
    ]


def test_loop(
    test_dataloader: DatasetDataLoader,
    module: nn.Module,
    loss: loss.Loss,
    metrics: list[Metric],
    device: Optional[torch.device],
    calculate_improvement: bool,
):
    module.eval()
    test_loss_sum = torch.zeros(1, device=device)
    num_metrics = len(metrics) * 2 if calculate_improvement else len(metrics)
    test_metrics_sum: torch.Tensor = torch.zeros(num_metrics, device=device)
    timer = CtxTimer()
    with torch.no_grad():
        mix: torch.Tensor
        target: torch.Tensor
        for mix, target in test_dataloader:
            mix = mix.to(device)
            target = target.to(device)
            separation: torch.Tensor = module(mix)
            separation_loss = torch.mean(loss(separation, target))
            metric_values = torch.stack(
                [torch.mean(metric(separation, target)) for metric in metrics]
                + _calculate_metric_improvement(metrics, mix, target, separation)
                if calculate_improvement
                else []
            )
            test_loss_sum += separation_loss
            test_metrics_sum += metric_values
    test_avg_loss = test_loss_sum.item() / len(test_dataloader)
    test_avg_metrics = test_metrics_sum / len(test_dataloader)
    return test_avg_loss, test_avg_metrics, timer.total


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    # Input dims: [Channel, Time]
    # Output dims: [Batch, Channel, Time]
    min_time = min([tensors[0].shape[-1] for tensors in batch])
    mixtures = torch.stack([tensors[0][:, :min_time] for tensors in batch])
    sources = torch.stack([tensors[1][:, :min_time] for tensors in batch])
    return mixtures, sources


def search_and_load_checkpoint(
    checkpointer: Checkpointer,
    model_name: str,
    model_config: ModelConfig,
    dataset_config: SpeechSeparationDatasetConfig,
):
    checkpoints = checkpointer.search_checkpoints(
        model_name,
        hidden_metadata={
            CheckpointerKeys.MODEL_CONFIG: model_config,
            CheckpointerKeys.DATASET_CONFIG: dataset_config,
        },
        desc_sort_by=Checkpointer.TIME_METADATA,
    )
    if len(checkpoints) > 0:
        checkpoint = checkpoints[0]
        return checkpoint, *checkpointer.load_checkpoint(checkpoint)
    return None


def save_checkpoint(
    checkpointer: Checkpointer,
    identifier: str,
    seed: int,
    model_name: str,
    epoch: int,
    train_avg_loss: float,
    valid_avg_loss: float,
    module: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    model_config: ModelConfig,
    dataset_config: SpeechSeparationDatasetConfig,
    optimization_configs: list[OptimizationConfig],
    train_config: TrainingConfig,
):
    checkpointer.save_checkpoint(
        model_name,
        visible_metadata=dict_any_to_str(
            {
                CheckpointerKeys.ID: identifier,
                CheckpointerKeys.EPOCH: epoch,
            }
        ),
        hidden_metadata={
            CheckpointerKeys.SEED: seed,
            CheckpointerKeys.TRAIN_AVERAGE_LOSS: train_avg_loss,
            CheckpointerKeys.VALID_AVERAGE_LOSS: valid_avg_loss,
            CheckpointerKeys.MODEL_CONFIG: model_config,
            CheckpointerKeys.DATASET_CONFIG: dataset_config,
            CheckpointerKeys.OPTIMIZATION_CONFIGS: optimization_configs,
            CheckpointerKeys.TRAINING_CONFIG: train_config,
        },
        data={
            CheckpointerKeys.MODULE_STATE_DICT: module.state_dict(),
            CheckpointerKeys.OPTIMIZER_STATE_DICT: optimizer.state_dict(),
            CheckpointerKeys.SCHEDULER_STATE_DICT: scheduler.state_dict(),
        },
    )


def sample_dataset(
    dataset: SpeechSeparationDatasetType, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return collate_fn([random.choice(dataset) for _ in range(batch_size)])


def _get_module_latency(
    module: nn.Module,
    dataset: SpeechSeparationDatasetType,
    batch_size: int,
    device: torch.device,
    number: int,
) -> float:
    total_time = 0
    for _ in range(number):
        # Start time
        start_time = time.perf_counter_ns()
        # Run module
        module(sample_dataset(dataset, batch_size)[0].to(device))
        # Synchronize
        synchronize_device(device)
        # Add time
        total_time += time.perf_counter_ns() - start_time
    return total_time / number


def _get_module_throughput(
    module: nn.Module,
    dataset: SpeechSeparationDatasetType,
    batch_size: int,
    device: torch.device,
    number: int,
) -> float:
    # Start time
    start_time = time.time()
    # Run module
    for _ in range(number):
        module(sample_dataset(dataset, batch_size)[0].to(device))
    # Synchronize
    synchronize_device(device)
    # Finish time
    total_time = time.time() - start_time
    # Return batch/s
    return number / total_time


def get_module_latency_and_throughput(
    module: nn.Module,
    dataset: SpeechSeparationDatasetType,
    batch_size: int,
    device: torch.device,
    warmup_number: int = 10,
    test_number: int = 10,
) -> tuple[float, float]:
    # Save current mode
    prev_mode = module.training
    # Switch to evaluation
    module.eval()
    # Use no gradients
    with torch.no_grad():
        # Initial execution
        module(sample_dataset(dataset, batch_size)[0].to(device))
        # Warmup phase
        for _ in range(warmup_number):
            module(sample_dataset(dataset, batch_size)[0].to(device))
        # Initial synchronize
        synchronize_device(device)
        # Test phase
        latency_time = _get_module_latency(
            module, dataset, batch_size, device, test_number
        )
        throughput_time = _get_module_throughput(
            module, dataset, batch_size, device, test_number
        )
    # Go back to the saved mode
    module.train(prev_mode)
    # Return time
    return latency_time, throughput_time
