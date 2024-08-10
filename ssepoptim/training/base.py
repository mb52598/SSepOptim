from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import ssepoptim.losses as losses
import ssepoptim.metrics as metrics
from ssepoptim.base.checkpointing import Checkpointer
from ssepoptim.base.configuration import BaseConfig
from ssepoptim.dataset import SpeechSeparationDatasetConfig
from ssepoptim.model import ModelConfig
from ssepoptim.optimization import OptimizationConfig
from ssepoptim.utils.context_timer import CtxTimer
from ssepoptim.utils.conversion import dict_any_to_str

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
    MODEL_STATE_DICT = "model_state_dict"
    OPTIMIZER_STATE_DICT = "optimizer_state_dict"
    SCHEDULER_STATE_DICT = "scheduler_state_dict"


class ReducedTrainingConfig(BaseConfig):
    epochs: int
    finetune_epochs: int
    batch_size: int
    lr: float
    shuffle: bool
    num_workers: int
    checkpoint_epoch_log: int
    loss: losses.Loss
    use_greedy_permutation_invariant_loss: bool
    load_last_checkpoint: bool
    apply_performance_optimizers: Optional[bool]
    test_only: Optional[bool]
    test_metrics: list[metrics.Metric]
    checkpoints_path: str
    distributed_training: Optional[bool]


class TrainingConfig(ReducedTrainingConfig):
    id: Optional[str]
    device: Optional[str]
    seed: Optional[int]


def train_loop(
    train_dataloader: DatasetDataLoader,
    model: nn.Module,
    loss: losses.Loss,
    optimizer: optim.Optimizer,
    device: Optional[torch.device],
):
    model.train()
    train_loss_sum = torch.zeros(1, device=device)
    timer = CtxTimer()
    mix: torch.Tensor
    target: torch.Tensor
    for mix, target in train_dataloader:
        mix = mix.to(device)
        target = target.to(device)
        separation = model(mix)
        separation_loss = torch.mean(loss(separation, target), dim=0)
        train_loss_sum += separation_loss
        optimizer.zero_grad(set_to_none=True)
        separation_loss.backward()
        optimizer.step()
    train_avg_loss = train_loss_sum.item() / len(train_dataloader)
    return train_avg_loss, timer.total


def valid_loop(
    valid_dataloader: DatasetDataLoader,
    model: nn.Module,
    loss: losses.Loss,
    device: Optional[torch.device],
):
    model.eval()
    valid_loss_sum = torch.zeros(1, device=device)
    timer = CtxTimer()
    with torch.no_grad():
        mix: torch.Tensor
        target: torch.Tensor
        for mix, target in valid_dataloader:
            mix = mix.to(device)
            target = target.to(device)
            separation = model(mix)
            separation_loss = torch.mean(loss(separation, target), dim=0)
            valid_loss_sum += separation_loss
    valid_avg_loss = valid_loss_sum.item() / len(valid_dataloader)
    return valid_avg_loss, timer.total


def test_loop(
    test_dataloader: DatasetDataLoader,
    model: nn.Module,
    loss: losses.Loss,
    metrics: list[metrics.Metric],
    device: Optional[torch.device],
):
    model.eval()
    test_loss_sum = torch.zeros(1, device=device)
    test_metrics_sum: torch.Tensor = torch.zeros(len(metrics), device=device)
    timer = CtxTimer()
    with torch.no_grad():
        mix: torch.Tensor
        target: torch.Tensor
        for mix, target in test_dataloader:
            mix = mix.to(device)
            target = target.to(device)
            separation = model(mix)
            separation_loss = torch.mean(loss(separation, target), dim=0)
            metric_values = torch.stack(
                [
                    torch.mean(metric(separation, target) - metric(mix, target))
                    for metric in metrics
                ]
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
    model: nn.Module,
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
            CheckpointerKeys.MODEL_STATE_DICT: model.state_dict(),
            CheckpointerKeys.OPTIMIZER_STATE_DICT: optimizer.state_dict(),
            CheckpointerKeys.SCHEDULER_STATE_DICT: scheduler.state_dict(),
        },
    )


def get_optimizer(model: nn.Module, lr: float):
    return optim.Adam(model.parameters(), lr)


def get_scheduler(optimizer: optim.Optimizer):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
