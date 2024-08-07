import logging
import random
import sys
from typing import Any, Optional
from uuid import uuid4

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import ssepoptim.losses as losses
import ssepoptim.metrics as metrics
from ssepoptim.base.checkpointing import Checkpointer
from ssepoptim.base.configuration import BaseConfig
from ssepoptim.dataset import (
    LenDataset,
    SpeechSeparationDataset,
    SpeechSeparationDatasetConfig,
    SpeechSeparationDatasetFactory,
)
from ssepoptim.model import ModelConfig, ModelFactory
from ssepoptim.optimization import (
    Optimization,
    OptimizationConfig,
    OptimizationFactory,
    Optimizations,
)
from ssepoptim.utils.context_timer import CtxTimer
from ssepoptim.utils.conversion import dict_any_to_str

logger = logging.getLogger(__name__)

_DatasetData = tuple[torch.Tensor, torch.Tensor]
_DataLoader = DataLoader[_DatasetData]


# Useful for preventing typing errors
class CheckpointerKeys:
    ID = "id"
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


class TrainingInferenceConfig(BaseConfig):
    id: Optional[str]
    epochs: int
    batch_size: int
    lr: float
    shuffle: bool
    num_workers: int
    checkpoint_epoch_log: int
    device: Optional[str]
    loss: losses.Loss
    load_last_checkpoint: bool
    seed: Optional[int]
    apply_performance_optimizers: Optional[bool]
    test_only: Optional[bool]
    test_metrics: list[metrics.Metric]
    checkpoints_path: str


def _train_loop(
    train_dataloader: _DataLoader,
    model: nn.Module,
    loss: losses.Loss,
    optimizer: optim.Optimizer,
    device: Optional[str],
):
    model.train()
    train_loss_sum = torch.zeros(1)
    timer = CtxTimer()
    mix: torch.Tensor
    target: torch.Tensor
    for mix, target in train_dataloader:
        mix = mix.to(device)
        target = target.to(device)
        separation = model(mix)
        separation_loss = torch.sum(loss(separation, target))
        train_loss_sum += separation_loss
        optimizer.zero_grad(set_to_none=True)
        separation_loss.backward()
        optimizer.step()
    train_avg_loss = train_loss_sum.item() / len(train_dataloader)
    return train_avg_loss, timer.total


def _valid_loop(
    valid_dataloader: _DataLoader,
    model: nn.Module,
    loss: losses.Loss,
    device: Optional[str],
):
    model.eval()
    valid_loss_sum = torch.zeros(1)
    timer = CtxTimer()
    with torch.no_grad():
        mix: torch.Tensor
        target: torch.Tensor
        for mix, target in valid_dataloader:
            mix = mix.to(device)
            target = target.to(device)
            separation = model(mix)
            separation_loss = torch.sum(loss(separation, target))
            valid_loss_sum += separation_loss
    valid_avg_loss = valid_loss_sum.item() / len(valid_dataloader)
    return valid_avg_loss, timer.total


def _test_loop(
    test_dataloader: _DataLoader,
    model: nn.Module,
    loss: losses.Loss,
    metrics: list[metrics.Metric],
    device: Optional[str],
):
    model.eval()
    test_loss_sum = torch.zeros(1)
    test_metrics_sum: torch.Tensor = torch.zeros([len(metrics)])
    timer = CtxTimer()
    with torch.no_grad():
        mix: torch.Tensor
        target: torch.Tensor
        for mix, target in test_dataloader:
            mix = mix.to(device)
            target = target.to(device)
            separation = model(mix)
            separation_loss = torch.sum(loss(separation, target))
            metric_values = torch.stack(
                [torch.sum(metric(separation, target)) for metric in metrics]
            )
            test_loss_sum += separation_loss
            test_metrics_sum += metric_values
    test_avg_loss = test_loss_sum.item() / len(test_dataloader)
    test_avg_metrics = test_metrics_sum / len(test_dataloader)
    return test_avg_loss, test_avg_metrics, timer.total


def _collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    # Input dims: [Channel, Time]
    # Output dims: [Batch, Channel, Time]
    min_time = min([tensors[0].shape[-1] for tensors in batch])
    mixtures = torch.stack([tensors[0][:, :min_time] for tensors in batch])
    sources = torch.stack([tensors[1][:, :min_time] for tensors in batch])
    return mixtures, sources


def _get_dataloader(
    dataset: LenDataset[_DatasetData], train_infer_config: TrainingInferenceConfig
) -> _DataLoader:
    pin_memory = train_infer_config["device"] != "cpu"
    return DataLoader(
        dataset,
        train_infer_config["batch_size"],
        train_infer_config["shuffle"],
        num_workers=train_infer_config["num_workers"],
        collate_fn=_collate_fn,
        pin_memory=pin_memory,
        generator=torch.Generator(device=train_infer_config["device"]),
    )


def _search_checkpoints(
    checkpointer: Checkpointer,
    model_name: str,
    model_config: ModelConfig,
    dataset_config: SpeechSeparationDatasetConfig,
    optimization_configs: list[OptimizationConfig],
    train_infer_config: TrainingInferenceConfig,
):
    checkpoints = checkpointer.search_checkpoints(
        model_name,
        hidden_metadata={
            CheckpointerKeys.MODEL_CONFIG: model_config,
            CheckpointerKeys.OPTIMIZATION_CONFIGS: optimization_configs,
            CheckpointerKeys.DATASET_CONFIG: dataset_config,
            CheckpointerKeys.TRAINING_CONFIG: train_infer_config,
        },
        desc_sort_by=Checkpointer.TIME_METADATA,
    )
    if len(checkpoints) > 0:
        checkpoint = checkpoints[0]
        return checkpoint, *checkpointer.load_checkpoint(checkpoint[0])
    return None


def _save_checkpoint(
    checkpointer: Checkpointer,
    identifier: str,
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
    train_infer_config: TrainingInferenceConfig,
):
    checkpointer.save_checkpoint(
        model_name,
        visible_metadata=dict_any_to_str(
            {
                CheckpointerKeys.ID: identifier,
                CheckpointerKeys.EPOCH: epoch,
                CheckpointerKeys.TRAIN_AVERAGE_LOSS: train_avg_loss,
                CheckpointerKeys.VALID_AVERAGE_LOSS: valid_avg_loss,
            }
        ),
        hidden_metadata={
            CheckpointerKeys.MODEL_CONFIG: model_config,
            CheckpointerKeys.DATASET_CONFIG: dataset_config,
            CheckpointerKeys.OPTIMIZATION_CONFIGS: optimization_configs,
            CheckpointerKeys.TRAINING_CONFIG: train_infer_config,
        },
        data={
            CheckpointerKeys.MODEL_STATE_DICT: model.state_dict(),
            CheckpointerKeys.OPTIMIZER_STATE_DICT: optimizer.state_dict(),
            CheckpointerKeys.SCHEDULER_STATE_DICT: scheduler.state_dict(),
        },
    )


def train(
    identifier: str,
    model_name: str,
    model: nn.Module,
    dataset: SpeechSeparationDataset,
    optimizations: list[Optimization],
    checkpointer: Checkpointer,
    checkpoint: tuple[str, dict[str, str], dict[str, Any], dict[str, Any]] | None,
    loss: losses.Loss,
    model_config: ModelConfig,
    dataset_config: SpeechSeparationDatasetConfig,
    optimization_configs: list[OptimizationConfig],
    train_infer_config: TrainingInferenceConfig,
):
    # Setup variables
    train_dataloader = _get_dataloader(dataset.get_train(), train_infer_config)
    valid_dataloader = _get_dataloader(dataset.get_valid(), train_infer_config)
    optimizer = optim.Adam(model.parameters(), train_infer_config["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    # Load checkpoint if configured
    start_epoch = 1
    if train_infer_config["load_last_checkpoint"]:
        if checkpoint is not None:
            checkpoint_name, visible_metadata, _, data = checkpoint
            start_epoch = int(visible_metadata[CheckpointerKeys.EPOCH])
            model.load_state_dict(data[CheckpointerKeys.MODEL_STATE_DICT])
            optimizer.load_state_dict(data[CheckpointerKeys.OPTIMIZER_STATE_DICT])
            scheduler.load_state_dict(data[CheckpointerKeys.SCHEDULER_STATE_DICT])
            logger.info("Using checkpoint for training: %s", checkpoint_name)
        else:
            logger.warn(
                "Unable to find any checkpoints for training, starting from the beginning"
            )
    # Apply optimizations and begin data loop
    model = Optimizations.apply(model, optimizations, stage="train")
    # Data loop
    timer = CtxTimer()
    for epoch in range(start_epoch, train_infer_config["epochs"] + 1):
        logger.info("Epoch %d", epoch)
        #
        train_avg_loss, train_time = _train_loop(
            train_dataloader, model, loss, optimizer, train_infer_config["device"]
        )
        logger.info("Train|Time: %f|Loss: %f", train_time, train_avg_loss)
        #
        valid_avg_loss, valid_time = _valid_loop(
            valid_dataloader, model, loss, train_infer_config["device"]
        )
        logger.info("Valid|Time: %f|Loss: %f", valid_time, valid_avg_loss)
        #
        logger.info(
            "Epoch %d|Time: %f|Loss: %f",
            epoch,
            train_time + valid_time,
            train_avg_loss + valid_avg_loss,
        )
        #
        if epoch % train_infer_config["checkpoint_epoch_log"] == 0:
            _save_checkpoint(
                checkpointer,
                identifier,
                model_name,
                epoch,
                train_avg_loss,
                valid_avg_loss,
                model,
                optimizer,
                scheduler,
                model_config,
                dataset_config,
                optimization_configs,
                train_infer_config,
            )
        #
        scheduler.step(train_avg_loss)
    # Log total time
    logger.info(
        "Training|Epochs: %d|Time: %f|Max cuda memory: %d",
        train_infer_config["epochs"],
        timer.total,
        torch.cuda.max_memory_allocated(train_infer_config["device"]),
    )
    torch.cuda.reset_peak_memory_stats(train_infer_config["device"])


def test(
    model: nn.Module,
    dataset: SpeechSeparationDataset,
    optimizations: list[Optimization],
    checkpoint: tuple[str, dict[str, str], dict[str, Any], dict[str, Any]] | None,
    loss: losses.Loss,
    train_infer_config: TrainingInferenceConfig,
):
    logger.info(
        "Using test metrics: %s",
        ", ".join(metric.__name__ for metric in train_infer_config["test_metrics"]),
    )
    # Setup variables
    test_dataloader = _get_dataloader(dataset.get_test(), train_infer_config)
    # Check if we need to load the model
    if train_infer_config["test_only"]:
        if checkpoint is None:
            raise RuntimeError("Unable to find checkpoint to test")
        checkpoint_name, _, _, data = checkpoint
        model.load_state_dict(data[CheckpointerKeys.MODEL_STATE_DICT])
        logger.info("Using checkpoint for testing: %s", checkpoint_name)
    # Apply optimizations and begin data loop
    model = Optimizations.apply(model, optimizations, stage="test")
    # Data loop
    test_avg_loss, test_avg_metrics, test_time = _test_loop(
        test_dataloader,
        model,
        loss,
        train_infer_config["test_metrics"],
        train_infer_config["device"],
    )
    # Log data
    metrics_str = ", ".join(["%f"] * len(test_avg_metrics))
    logger.info(
        f"Test|Time: %f|Loss: %f|Metrics: {metrics_str}|Max cuda memory: %d",
        test_time,
        test_avg_loss,
        *test_avg_metrics,
        torch.cuda.max_memory_allocated(train_infer_config["device"]),
    )
    torch.cuda.reset_peak_memory_stats(train_infer_config["device"])


def train_test(
    model_name: str,
    dataset_name: str,
    optimization_names: list[str],
    model_config: ModelConfig,
    dataset_config: SpeechSeparationDatasetConfig,
    optimization_configs: list[OptimizationConfig],
    train_infer_config: TrainingInferenceConfig,
) -> nn.Module:
    # Setup default device
    if train_infer_config["device"] is not None:
        torch.set_default_device(train_infer_config["device"])
    # Setup seed
    if train_infer_config["seed"] is not None:
        seed = train_infer_config["seed"]
    else:
        seed = random.randrange(sys.maxsize)
    random.seed(seed)
    torch.manual_seed(seed)
    # Apply performance optimizers
    if train_infer_config["apply_performance_optimizers"]:
        torch.jit.enable_onednn_fusion(True)
        torch.backends.cudnn.benchmark = True
    # Setup variables
    model = ModelFactory.get_object(model_name, model_config)
    dataset = SpeechSeparationDatasetFactory.get_object(dataset_name, dataset_config)
    optimizations = [
        OptimizationFactory.get_object(optimization_name, optimization_config)
        for optimization_name, optimization_config in zip(
            optimization_names, optimization_configs
        )
    ]
    checkpointer = Checkpointer(
        train_infer_config["checkpoints_path"], train_infer_config["device"]
    )
    loss = losses.create_permutation_invariant_loss(train_infer_config["loss"])
    # Load checkpoint
    checkpoint = None
    if train_infer_config["load_last_checkpoint"] or train_infer_config["test_only"]:
        checkpoint = _search_checkpoints(
            checkpointer,
            model_name,
            model_config,
            dataset_config,
            optimization_configs,
            train_infer_config,
        )
        if checkpoint is not None:
            _, visible_metadata, _, _ = checkpoint
            train_infer_config["id"] = visible_metadata[CheckpointerKeys.ID]
            logger.info("Using checkpoint: %s", checkpoint)
    # Load id
    if train_infer_config["id"] is None:
        identifier = str(uuid4())
    else:
        identifier = train_infer_config["id"]
    # Log variables
    logger.info("Using id: %s", identifier)
    logger.info("Using seed: %d", seed)
    logger.info("Using loss: %s", train_infer_config["loss"].__name__)
    # Train (if enabled)
    if not train_infer_config["test_only"]:
        train(
            identifier,
            model_name,
            model,
            dataset,
            optimizations,
            checkpointer,
            checkpoint,
            loss,
            model_config,
            dataset_config,
            optimization_configs,
            train_infer_config,
        )
    # Test
    test(
        model,
        dataset,
        optimizations,
        checkpoint,
        loss,
        train_infer_config,
    )
    # Return the trained model
    return model
