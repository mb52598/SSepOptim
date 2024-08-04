import logging
import random
from typing import Optional
from uuid import uuid4

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import ssepoptim.metrics as metrics
from ssepoptim.base.checkpointing import Checkpointer, CheckpointerConfig
from ssepoptim.base.configuration import BaseConfig
from ssepoptim.dataset import (
    SpeechSeparationDatasetConfig,
    SpeechSeparationDatasetFactory,
)
from ssepoptim.model import ModelConfig, ModelFactory
from ssepoptim.optimization import (
    OptimizationConfig,
    OptimizationFactory,
    Optimizations,
)
from ssepoptim.utils.context_timer import CtxTimer
from ssepoptim.utils.conversion import dict_any_to_str

logger = logging.getLogger(__name__)

_DataLoader = DataLoader[tuple[torch.Tensor, torch.Tensor]]


# Useful for preventing typing errors
class CheckpointerKeys:
    ID = "id"
    EPOCH = "epoch"
    TRAIN_AVERAGE_LOSS = "train_avg_loss"
    VALID_AVERAGE_LOSS = "valid_avg_loss"
    MODEL_CONFIG = "model_config"
    OPTIMIZATION_CONFIGS = "optimization_configs"
    DATASET_CONFIG = "dataset_config"
    CHECKPOINTER_CONFIG = "checkpointer_config"
    TRAINING_CONFIG = "training_config"
    MODEL_STATE_DICT = "model_state_dict"
    OPTIMIZER_STATE_DICT = "optimizer_state_dict"
    SCHEDULER_STATE_DICT = "scheduler_state_dict"


class TrainingInferenceConfig(BaseConfig):
    id: Optional[str]
    epochs: int
    batch_size: int
    lr: float
    number_of_speakers: int
    shuffle: bool
    num_workers: int
    checkpoint_epoch_log: int
    device: Optional[str]
    metric: metrics.Metric
    load_last_checkpoint: bool
    seed: Optional[int]


def _train_loop(
    train_dataloader: _DataLoader,
    model: nn.Module,
    metric: metrics.Metric,
    optimizer: optim.Optimizer,
):
    model.train()
    train_loss_sum: torch.Tensor = 0.0
    timer = CtxTimer()
    for mix, target in train_dataloader:
        separation = model(mix)
        loss = -torch.sum(metric(separation, target))
        train_loss_sum += loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    train_avg_loss = train_loss_sum.sum().item() / len(train_dataloader)
    return train_avg_loss, timer.total


def _valid_loop(
    valid_dataloader: _DataLoader,
    model: nn.Module,
    metric: metrics.Metric,
):
    model.eval()
    valid_loss_sum: torch.Tensor = 0.0
    timer = CtxTimer()
    with torch.no_grad():
        for mix, target in valid_dataloader:
            separation = model(mix)
            loss = -torch.sum(metric(separation, target))
            valid_loss_sum += loss
    valid_avg_loss = valid_loss_sum.sum().item() / len(valid_dataloader)
    return valid_avg_loss, timer.total


def train(
    model_name: str,
    dataset_name: str,
    optimization_names: list[str],
    model_config: ModelConfig,
    dataset_config: SpeechSeparationDatasetConfig,
    optimization_configs: list[OptimizationConfig],
    checkpointer_config: CheckpointerConfig,
    train_infer_config: TrainingInferenceConfig,
) -> nn.Module:
    # Setup default device
    if train_infer_config["device"] is not None:
        torch.set_default_device(train_infer_config["device"])
    # Setup manual seed
    if train_infer_config["seed"] is not None:
        random.seed(train_infer_config["seed"])
        torch.manual_seed(train_infer_config["seed"])
    # Setup variables
    model = ModelFactory.get_object(model_name, model_config)
    dataset = SpeechSeparationDatasetFactory.get_object(dataset_name, dataset_config)
    optimizations = [
        OptimizationFactory.get_object(optimization_name, optimization_config)
        for optimization_name, optimization_config in zip(
            optimization_names, optimization_configs
        )
    ]
    metric = train_infer_config["metric"]
    train_dataset = dataset.get_train()
    valid_dataset = dataset.get_valid()
    train_dataloader = DataLoader(
        train_dataset,
        train_infer_config["batch_size"],
        train_infer_config["shuffle"],
        num_workers=train_infer_config["num_workers"],
        generator=torch.Generator(device=train_infer_config["device"]),
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        train_infer_config["batch_size"],
        train_infer_config["shuffle"],
        num_workers=train_infer_config["num_workers"],
        generator=torch.Generator(device=train_infer_config["device"]),
    )
    optimizer = optim.Adam(model.parameters(), train_infer_config["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    checkpointer = Checkpointer(checkpointer_config)
    # Load checkpoint if configured
    start_epoch = 1
    if train_infer_config["load_last_checkpoint"]:
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
            logger.info("Using checkpoint: %s", checkpoint)
            visible_metadata, _, data = checkpointer.load_checkpoint(checkpoint[0])
            train_infer_config["id"] = visible_metadata[CheckpointerKeys.ID]
            start_epoch = int(visible_metadata[CheckpointerKeys.EPOCH])
            model.load_state_dict(data[CheckpointerKeys.MODEL_STATE_DICT])
            optimizer.load_state_dict(data[CheckpointerKeys.OPTIMIZER_STATE_DICT])
            scheduler.load_state_dict(data[CheckpointerKeys.SCHEDULER_STATE_DICT])
        else:
            logger.warn("Unable to find any checkpoints, starting from the beginning")
    # Create id if not available
    if train_infer_config["id"] is None:
        train_infer_config["id"] = str(uuid4())
    # Print id
    logger.info("Using id: %s", train_infer_config["id"])
    # Apply optimizations and begin data loop
    model = Optimizations.apply(model, optimizations, stage="train")
    # Data loop
    timer = CtxTimer()
    for epoch in range(start_epoch, train_infer_config["epochs"] + 1):
        logger.info("Epoch %d", epoch)
        #
        train_avg_loss, train_time = _train_loop(
            train_dataloader, model, metric, optimizer
        )
        logger.info("Train|Time: %f|Loss: %f", train_time, train_avg_loss)
        #
        valid_avg_loss, valid_time = _valid_loop(valid_dataloader, model, metric)
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
            checkpointer.save_checkpoint(
                model_name,
                visible_metadata=dict_any_to_str(
                    {
                        CheckpointerKeys.ID: train_infer_config["id"],
                        CheckpointerKeys.EPOCH: epoch,
                        CheckpointerKeys.TRAIN_AVERAGE_LOSS: train_avg_loss,
                        CheckpointerKeys.VALID_AVERAGE_LOSS: valid_avg_loss,
                    }
                ),
                hidden_metadata={
                    CheckpointerKeys.MODEL_CONFIG: model_config,
                    CheckpointerKeys.DATASET_CONFIG: dataset_config,
                    CheckpointerKeys.OPTIMIZATION_CONFIGS: optimization_configs,
                    CheckpointerKeys.CHECKPOINTER_CONFIG: checkpointer_config,
                    CheckpointerKeys.TRAINING_CONFIG: train_infer_config,
                },
                data={
                    CheckpointerKeys.MODEL_STATE_DICT: model.state_dict(),
                    CheckpointerKeys.OPTIMIZER_STATE_DICT: optimizer.state_dict(),
                    CheckpointerKeys.SCHEDULER_STATE_DICT: scheduler.state_dict(),
                },
            )
        #
        scheduler.step(train_avg_loss)
    # Log total time
    logger.info("Training|Epochs: %d|Time: %f", train_infer_config["epochs"], timer.total)
    # Return the trained model
    return model
