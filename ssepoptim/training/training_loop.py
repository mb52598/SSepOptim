import logging
import random
from typing import Optional, cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import ssepoptim.losses as losses
from ssepoptim.base.checkpointing import Checkpointer
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
from ssepoptim.training.base import (
    CheckpointerKeys,
    DatasetData,
    DatasetDataLoader,
    ReducedTrainingConfig,
    TrainingConfig,
    collate_fn,
    get_optimizer,
    get_scheduler,
    save_checkpoint,
    test_loop,
    train_loop,
    valid_loop,
)
from ssepoptim.utils.context_timer import CtxTimer
from ssepoptim.utils.distributed import get_global_rank

logger = logging.getLogger(__name__)


def _get_dataloader(
    dataset: LenDataset[DatasetData],
    device: torch.device,
    train_config: ReducedTrainingConfig,
) -> DatasetDataLoader:
    if train_config["distributed_training"]:
        return DataLoader(
            dataset,
            batch_size=train_config["batch_size"],
            shuffle=False,
            sampler=DistributedSampler(dataset),
            num_workers=train_config["num_workers"],
            collate_fn=collate_fn,
            pin_memory=True,
        )
    else:
        pin_memory = device.type != "cpu"
        return DataLoader(
            dataset,
            batch_size=train_config["batch_size"],
            shuffle=train_config["shuffle"],
            num_workers=train_config["num_workers"],
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            generator=torch.Generator(device=device),
        )


class CheckpointSaver:
    def __init__(
        self,
        identifier: str,
        seed: int,
        model_name: str,
        model_config: ModelConfig,
        dataset_config: SpeechSeparationDatasetConfig,
        optimization_configs: list[OptimizationConfig],
        train_config: TrainingConfig,
    ):
        self._identifier = identifier
        self._seed = seed
        self._model_name = model_name
        self._model_config = model_config
        self._dataset_config = dataset_config
        self._optimization_configs = optimization_configs
        self._train_config = train_config

    def save_checkpoint(
        self,
        checkpointer: Checkpointer,
        epoch: int,
        train_avg_loss: float,
        valid_avg_loss: float,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
    ):
        save_checkpoint(
            checkpointer,
            self._identifier,
            self._seed,
            self._model_name,
            epoch,
            train_avg_loss,
            valid_avg_loss,
            model,
            optimizer,
            scheduler,
            self._model_config,
            self._dataset_config,
            self._optimization_configs,
            self._train_config,
        )


def _train(
    model: nn.Module,
    dataset: SpeechSeparationDataset,
    optimizations: list[Optimization],
    checkpointer: Checkpointer,
    checkpoint_name: Optional[str],
    checkpoint_saver: CheckpointSaver,
    loss: losses.Loss,
    device: torch.device,
    train_config: ReducedTrainingConfig,
) -> nn.Module:
    # Setup variables
    train_dataloader = _get_dataloader(dataset.get_train(), device, train_config)
    valid_dataloader = _get_dataloader(dataset.get_valid(), device, train_config)
    optimizer = get_optimizer(model, train_config["lr"])
    scheduler = get_scheduler(optimizer)
    # Load checkpoint if configured
    start_epoch = 1
    if train_config["load_last_checkpoint"]:
        if checkpoint_name is not None:
            visible_metadata, _, data = checkpointer.load_checkpoint(checkpoint_name)
            start_epoch = int(visible_metadata[CheckpointerKeys.EPOCH]) + 1
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
    # Transfer the model to the device
    model = model.to(device)
    # If we are distributed wrap the model in DDP
    rank = None
    if train_config["distributed_training"]:
        rank = get_global_rank()
        model = DDP(model, device_ids=[device.index])
    # Data loop
    timer = CtxTimer()
    for epoch in range(start_epoch, train_config["epochs"] + 1):
        logger.info("Epoch %d", epoch)
        #
        if train_config["distributed_training"]:
            cast(DistributedSampler[DatasetData], train_dataloader.sampler).set_epoch(
                epoch
            )
        #
        train_avg_loss, train_time = train_loop(
            train_dataloader, model, loss, optimizer, device
        )
        logger.info("Train|Time: %f|Loss: %f", train_time, train_avg_loss)
        #
        valid_avg_loss, valid_time = valid_loop(valid_dataloader, model, loss, device)
        logger.info("Valid|Time: %f|Loss: %f", valid_time, valid_avg_loss)
        #
        logger.info(
            "Epoch %d|Time: %f|Loss: %f",
            epoch,
            train_time + valid_time,
            train_avg_loss + valid_avg_loss,
        )
        #
        if epoch % train_config["checkpoint_epoch_log"] == 0:
            if not train_config["distributed_training"] or rank == 0:
                checkpoint_saver.save_checkpoint(
                    checkpointer,
                    epoch,
                    train_avg_loss,
                    valid_avg_loss,
                    model,
                    optimizer,
                    scheduler,
                )
        #
        scheduler.step(train_avg_loss)
    # Log total time
    logger.info(
        "Training|Epochs: %d|Time: %f|Max cuda memory: %d",
        train_config["epochs"],
        timer.total,
        torch.cuda.max_memory_allocated(device),
    )
    torch.cuda.reset_peak_memory_stats(device)
    # Return model(unwrapped if distributed)
    if train_config["distributed_training"]:
        return model.module
    else:
        return model


def _fine_tune(
    model: nn.Module,
    dataset: SpeechSeparationDataset,
    optimizations: list[Optimization],
    checkpointer: Checkpointer,
    checkpoint_saver: CheckpointSaver,
    loss: losses.Loss,
    device: torch.device,
    train_config: ReducedTrainingConfig,
) -> nn.Module:
    # Setup variables
    train_dataloader = _get_dataloader(dataset.get_train(), device, train_config)
    optimizer = get_optimizer(model, train_config["lr"])
    scheduler = get_scheduler(optimizer)
    # Apply optimizations and begin data loop
    model = Optimizations.apply(model, optimizations, stage="train")
    # Transfer the model to the device
    model = model.to(device)
    # If we are distributed wrap the model in DDP
    rank = None
    if train_config["distributed_training"]:
        rank = get_global_rank()
        model = DDP(model, device_ids=[device.index])
    # Data loop
    timer = CtxTimer()
    for epoch in range(1, train_config["finetune_epochs"] + 1):
        logger.info("Fine-Tune|Epoch %d", epoch)
        #
        if train_config["distributed_training"]:
            cast(DistributedSampler[DatasetData], train_dataloader.sampler).set_epoch(
                epoch
            )
        #
        train_avg_loss, train_time = train_loop(
            train_dataloader, model, loss, optimizer, device
        )
        logger.info(
            "Fine-Tune|Epoch %d|Time: %f|Loss: %f", epoch, train_time, train_avg_loss
        )
        #
        scheduler.step(train_avg_loss)
    # Save model checkpoint
    if not train_config["distributed_training"] or rank == 0:
        checkpoint_saver.save_checkpoint(
            checkpointer,
            train_config["epochs"] + train_config["finetune_epochs"],
            0,
            0,
            model,
            optimizer,
            scheduler,
        )
    # Log total time
    logger.info(
        "Fine-Tune|Epochs: %d|Time: %f|Max cuda memory: %d",
        train_config["finetune_epochs"],
        timer.total,
        torch.cuda.max_memory_allocated(device),
    )
    torch.cuda.reset_peak_memory_stats(device)
    # Return model(unwrapped if distributed)
    if train_config["distributed_training"]:
        return model.module
    else:
        return model


def _test(
    model: nn.Module,
    dataset: SpeechSeparationDataset,
    optimizations: list[Optimization],
    checkpointer: Checkpointer,
    checkpoint_name: Optional[str],
    loss: losses.Loss,
    device: torch.device,
    train_config: ReducedTrainingConfig,
):
    logger.info(
        "Using test metrics: %s",
        ", ".join(metric.__name__ for metric in train_config["test_metrics"]),
    )
    # Setup variables
    test_dataloader = _get_dataloader(dataset.get_test(), device, train_config)
    # If we didn't train we need to load a checkpoint
    if train_config["test_only"]:
        assert checkpoint_name is not None
        _, _, data = checkpointer.load_checkpoint(checkpoint_name)
        model.load_state_dict(data[CheckpointerKeys.MODEL_STATE_DICT])
        logger.info("Using checkpoint for testing: %s", checkpoint_name)
    # Apply optimizations and begin data loop
    model = Optimizations.apply(model, optimizations, stage="test")
    # Transfer the model to the device
    model = model.to(device)
    # If we are distributed wrap the model in DDP
    if train_config["distributed_training"]:
        model = DDP(model, device_ids=[device.index])
    # Data loop
    test_avg_loss, test_avg_metrics, test_time = test_loop(
        test_dataloader,
        model,
        loss,
        train_config["test_metrics"],
        device,
    )
    # Log data
    metrics_str = ", ".join(["%f"] * len(test_avg_metrics))
    logger.info(
        f"Test|Time: %f|Loss: %f|Metrics: {metrics_str}|Max cuda memory: %d",
        test_time,
        test_avg_loss,
        *test_avg_metrics,
        torch.cuda.max_memory_allocated(device),
    )
    torch.cuda.reset_peak_memory_stats(device)


def train_test(
    identifier: str,
    seed: int,
    checkpoint_name: Optional[str],
    device: torch.device,
    model_name: str,
    dataset_name: str,
    optimization_names: list[str],
    model_config: ModelConfig,
    dataset_config: SpeechSeparationDatasetConfig,
    optimization_configs: list[OptimizationConfig],
    train_config: TrainingConfig,
):
    # Setup seed
    random.seed(seed)
    torch.manual_seed(seed)
    # Apply performance optimizers
    if train_config["apply_performance_optimizers"]:
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
    checkpointer = Checkpointer(train_config["checkpoints_path"], device)
    checkpoint_saver = CheckpointSaver(
        identifier,
        seed,
        model_name,
        model_config,
        dataset_config,
        optimization_configs,
        train_config,
    )
    loss = losses.PermutationInvariantLoss(
        train_config["loss"],
        train_config["use_greedy_permutation_invariant_loss"],
    )
    # Log variables
    logger.info("Using id: %s", identifier)
    logger.info("Using seed: %d", seed)
    logger.info("Using loss: %s", train_config["loss"].__name__)
    # Train (if enabled)
    if not train_config["test_only"]:
        model = _train(
            model,
            dataset,
            optimizations,
            checkpointer,
            checkpoint_name,
            checkpoint_saver,
            loss,
            device,
            train_config,
        )
        if Optimizations.requireFinetune(optimizations):
            model = _fine_tune(
                model,
                dataset,
                optimizations,
                checkpointer,
                checkpoint_saver,
                loss,
                device,
                train_config,
            )
    # Test
    _test(
        model,
        dataset,
        optimizations,
        checkpointer,
        checkpoint_name,
        loss,
        device,
        train_config,
    )
