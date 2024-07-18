import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import ssepoptim.metrics as metrics
from ssepoptim.base.checkpointing import Checkpointer, CheckpointerConfig
from ssepoptim.base.configuration import BaseConfig
from ssepoptim.dataset import (
    SpeechSeparationDatasetConfig,
    SpeechSeparationDatasetFactory,
)
from ssepoptim.model import ModelConfig, ModelFactory
from ssepoptim.optimization import OptimizationConfig, OptimizationFactory
from ssepoptim.utils.context_helper import zip_context
from ssepoptim.utils.conversion import dict_any_to_str

logger = logging.getLogger(__name__)

MODEL_STATE_DICT_STR = "model_state_dict"
OPTIMIZER_STATE_DICT_STR = "optmizer_state_dict"
SCHEDULER_STATE_DICT_STR = "scheduler_state_dict"

_DataLoader = DataLoader[tuple[torch.Tensor, torch.Tensor]]


class TrainingConfig(BaseConfig):
    epochs: int
    batch_size: int
    lr: float
    number_of_speakers: int
    shuffle: bool
    checkpoint_epoch_log: int
    device: Optional[str]
    metric: metrics.Metric
    load_last_checkpoint: bool


def _train_loop(
    epoch: int,
    model_name: str,
    train_dataloader: _DataLoader,
    valid_dataloader: _DataLoader,
    model: nn.Module,
    metric: metrics.Metric,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    checkpointer: Checkpointer,
    model_config: ModelConfig,
    dataset_config: SpeechSeparationDatasetConfig,
    optimization_configs: list[OptimizationConfig],
    checkpointer_config: CheckpointerConfig,
    training_config: TrainingConfig,
):
    model.train()
    train_loss_sum: torch.Tensor = 0.0
    for mix, target in train_dataloader:
        separation = model(mix)
        loss = -metric(separation, target)
        train_loss_sum += loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if epoch % training_config["checkpoint_epoch_log"] == 0:
        model.eval()
        valid_loss_sum: torch.Tensor = 0.0
        with torch.no_grad():
            for mix, target in valid_dataloader:
                separation = model(mix)
                loss = -metric(separation, target)
                valid_loss_sum += loss
        train_avg_loss = train_loss_sum.sum().item() / len(train_dataloader)
        valid_avg_loss = valid_loss_sum.sum().item() / len(valid_dataloader)
        logger.info(
            "Train SI-SNR: %f, Valid SI-SNR: %f", train_avg_loss, valid_avg_loss
        )
        checkpointer.save(
            model_name,
            major_metadata=dict_any_to_str(
                {
                    "epoch": epoch,
                    "train_avg_loss": train_avg_loss,
                    "valid_avg_loss": valid_avg_loss,
                    **model_config,
                    **dataset_config,
                    **{
                        k: v
                        for config in optimization_configs
                        for k, v in config.items()
                    },
                    **checkpointer_config,
                    **training_config,
                }
            ),
            minor_metadata={
                MODEL_STATE_DICT_STR: model.state_dict(),
                OPTIMIZER_STATE_DICT_STR: optimizer.state_dict(),
                SCHEDULER_STATE_DICT_STR: scheduler.state_dict(),
            },
        )
    scheduler.step(train_loss_sum)


def train(
    model_name: str,
    dataset_name: str,
    optimization_names: list[str],
    model_config: ModelConfig,
    dataset_config: SpeechSeparationDatasetConfig,
    optimization_configs: list[OptimizationConfig],
    checkpointer_config: CheckpointerConfig,
    training_config: TrainingConfig,
):
    # Log configuration
    logger.info("Using model: %s", model_name)
    logger.info("Using dataset: %s", dataset_name)
    logger.info(
        "Using optimizations: %s",
        ", ".join(optimization_names) if len(optimization_names) > 0 else "none",
    )
    logger.info("Using metric: %s", training_config["metric"].__name__)
    # Setup variables
    model = ModelFactory.get_object(model_name, model_config)
    dataset = SpeechSeparationDatasetFactory.get_object(dataset_name, dataset_config)
    optimizations = [
        OptimizationFactory.get_object(optimization_name, optimization_config)
        for optimization_name, optimization_config in zip(
            optimization_names, optimization_configs
        )
    ]
    metric = training_config["metric"]
    train_dataset = dataset.get_train()
    valid_dataset = dataset.get_valid()
    train_dataloader = DataLoader(
        train_dataset, training_config["batch_size"], training_config["shuffle"]
    )
    valid_dataloader = DataLoader(
        valid_dataset, training_config["batch_size"], training_config["shuffle"]
    )
    optimizer = optim.Adam(model.parameters(), training_config["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    checkpointer = Checkpointer(checkpointer_config)
    # Load checkpoint if configured
    if training_config["load_last_checkpoint"]:
        checkpoints = checkpointer.search_checkpoints(
            model_name,
            dict_any_to_str({**model_config, **dataset_config, **training_config}),
            Checkpointer.TIME_METADATA,
        )
        if len(checkpoints) > 0:
            checkpoint = checkpoints[0]
            logger.info("Using checkpoint: %s", checkpoint)
            minor_metadata = checkpointer.get_minor_metadata(checkpoint)
            model.load_state_dict(minor_metadata[MODEL_STATE_DICT_STR])
            optimizer.load_state_dict(minor_metadata[OPTIMIZER_STATE_DICT_STR])
            scheduler.load_state_dict(minor_metadata[SCHEDULER_STATE_DICT_STR])
    # Apply optimizations and begin data loop
    with zip_context(optimization.apply(model) for optimization in optimizations):
        for epoch in tqdm(range(training_config["epochs"])):
            _train_loop(
                epoch,
                model_name,
                train_dataloader,
                valid_dataloader,
                model,
                metric,
                optimizer,
                scheduler,
                checkpointer,
                model_config,
                dataset_config,
                optimization_configs,
                checkpointer_config,
                training_config,
            )
