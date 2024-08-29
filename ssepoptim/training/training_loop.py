import logging
import random
from typing import Optional, cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import ssepoptim.loss as loss
from ssepoptim.base.checkpointing import Checkpointer
from ssepoptim.dataset import (
    LenDataset,
    SpeechSeparationDatasetConfig,
    SpeechSeparationDatasetFactory,
    SpeechSeparationDatasetType,
)
from ssepoptim.metrics.pit import PermutationInvariantMetric
from ssepoptim.model import Model, ModelConfig, ModelFactory
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
    save_checkpoint,
    test_loop,
    train_loop,
    valid_loop,
)
from ssepoptim.training.early_stop import DummyEarlyStop, EarlyStop
from ssepoptim.training.training_observer import TrainingObservers
from ssepoptim.utils.distributed import get_global_rank, is_distributed

logger = logging.getLogger(__name__)


def _get_dataloader(
    dataset: LenDataset[DatasetData],
    device: torch.device,
    seed: int,
    train_config: ReducedTrainingConfig,
) -> DatasetDataLoader:
    if is_distributed():
        return DataLoader(
            dataset,
            batch_size=train_config["batch_size"],
            shuffle=False,
            sampler=DistributedSampler(dataset, seed=seed),
            num_workers=train_config["num_workers"],
            collate_fn=collate_fn,
            pin_memory=True,
            generator=torch.Generator(device=device),
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


def _scheduler_step(scheduler: optim.lr_scheduler.LRScheduler, train_avg_loss: float):
    if type(scheduler) is optim.lr_scheduler.ReduceLROnPlateau:
        scheduler.step(train_avg_loss)
    else:
        scheduler.step()


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
        module: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
    ):
        # Unwrap module if DDP
        if type(module) is DDP:
            module = cast(nn.Module, module.module)
        save_checkpoint(
            checkpointer,
            self._identifier,
            self._seed,
            self._model_name,
            epoch,
            train_avg_loss,
            valid_avg_loss,
            module,
            optimizer,
            scheduler,
            self._model_config,
            self._dataset_config,
            self._optimization_configs,
            self._train_config,
        )


def _load_train_checkpoint(
    checkpointer: Checkpointer,
    checkpoint_name: str,
    module: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
):
    visible_metadata, _, data = checkpointer.load_checkpoint(checkpoint_name)
    start_epoch = int(visible_metadata[CheckpointerKeys.EPOCH]) + 1
    module.load_state_dict(data[CheckpointerKeys.MODULE_STATE_DICT])
    optimizer.load_state_dict(data[CheckpointerKeys.OPTIMIZER_STATE_DICT])
    scheduler.load_state_dict(data[CheckpointerKeys.SCHEDULER_STATE_DICT])
    return start_epoch


def _load_test_checkpoint(
    checkpointer: Checkpointer,
    checkpoint_name: str,
    module: nn.Module,
):
    _, _, data = checkpointer.load_checkpoint(checkpoint_name)
    module.load_state_dict(data[CheckpointerKeys.MODULE_STATE_DICT])


def _train(
    module: nn.Module,
    model: Model,
    train_dataset: SpeechSeparationDatasetType,
    valid_dataset: SpeechSeparationDatasetType,
    test_dataset: SpeechSeparationDatasetType,
    optimizations: list[Optimization],
    checkpointer: Checkpointer,
    checkpoint_name: Optional[str],
    checkpoint_saver: CheckpointSaver,
    observers: TrainingObservers,
    loss: loss.Loss,
    device: torch.device,
    seed: int,
    train_config: ReducedTrainingConfig,
) -> nn.Module:
    # Setup variables
    train_dataloader = _get_dataloader(train_dataset, device, seed, train_config)
    valid_dataloader = _get_dataloader(valid_dataset, device, seed, train_config)
    optimizer = model.get_optimizer(module)
    scheduler = model.get_scheduler(optimizer)
    early_stop = cast(EarlyStop | None, train_config["early_stop"]) or DummyEarlyStop()
    # Load checkpoint if configured
    start_epoch = 1
    if train_config["load_last_checkpoint"]:
        if checkpoint_name is not None:
            start_epoch = _load_train_checkpoint(
                checkpointer, checkpoint_name, module, optimizer, scheduler
            )
            logger.info("Using checkpoint for training: %s", checkpoint_name)
        else:
            logger.warn(
                "Unable to find any checkpoints for training, starting from the beginning"
            )
    # Apply optimizations
    module = Optimizations.apply(module, optimizations, "TRAIN_START", locals())
    # Transfer the module to the device
    module = module.to(device)
    # Activate observer
    observers.on_training_start(locals())
    # If we are distributed wrap the module in DDP
    rank = None
    if is_distributed():
        rank = get_global_rank()
        module = DDP(
            module,
            device_ids=[device.index],
            find_unused_parameters=train_config["distributed_find_unused_params"],
        )
    # Data loop
    for epoch in range(start_epoch, train_config["epochs"] + 1):
        observers.on_training_epoch_start(locals())
        #
        logger.info("Epoch %d", epoch)
        #
        if type(train_dataloader.sampler) is DistributedSampler:
            train_dataloader.sampler.set_epoch(epoch)
        #
        train_avg_loss, train_time = train_loop(
            train_dataloader,
            module,
            loss,
            optimizer,
            device,
            train_config["gradient_accumulation_steps"],
            train_config["clip_grad_norm"],
        )
        logger.info("Train|Time: %f s|Loss: %f", train_time, train_avg_loss)
        #
        observers.on_training_epoch_validation(locals())
        #
        valid_avg_loss, valid_time = valid_loop(valid_dataloader, module, loss, device)
        logger.info("Valid|Time: %f s|Loss: %f", valid_time, valid_avg_loss)
        #
        logger.info(
            "Epoch %d|Time: %f s|Loss: %f",
            epoch,
            train_time + valid_time,
            train_avg_loss + valid_avg_loss,
        )
        #
        if epoch % train_config["checkpoint_epoch_log"] == 0:
            if not is_distributed() or rank == 0:
                checkpoint_saver.save_checkpoint(
                    checkpointer,
                    epoch,
                    train_avg_loss,
                    valid_avg_loss,
                    module,
                    optimizer,
                    scheduler,
                )
        #
        _scheduler_step(scheduler, train_avg_loss)
        #
        observers.on_training_epoch_end(locals())
        #
        if early_stop.should_early_stop(locals()):
            logger.info("Early stopping on epoch %d", epoch)
            break
    # Unwrap module if distributed
    if type(module) is DDP:
        module = cast(nn.Module, module.module)
    # Apply optimizations
    module = Optimizations.apply(module, optimizations, "TRAIN_END", locals())
    # Activate observer
    observers.on_training_end(locals())
    # Return module
    return module


def _fine_tune(
    module: nn.Module,
    model: Model,
    train_dataset: SpeechSeparationDatasetType,
    valid_dataset: SpeechSeparationDatasetType,
    test_dataset: SpeechSeparationDatasetType,
    optimizations: list[Optimization],
    checkpointer: Checkpointer,
    checkpoint_saver: CheckpointSaver,
    observers: TrainingObservers,
    loss: loss.Loss,
    device: torch.device,
    seed: int,
    train_config: ReducedTrainingConfig,
) -> nn.Module:
    # Setup variables
    train_dataloader = _get_dataloader(train_dataset, device, seed, train_config)
    valid_dataloader = _get_dataloader(valid_dataset, device, seed, train_config)
    optimizer = model.get_optimizer(module)
    scheduler = model.get_scheduler(optimizer)
    # Apply optimizations
    module = Optimizations.apply(module, optimizations, "FINETUNE_START", locals())
    # Transfer the module to the device
    module = module.to(device)
    # Activate observer
    observers.on_fine_tuning_start(locals())
    # If we are distributed wrap the module in DDP
    rank = None
    if is_distributed():
        rank = get_global_rank()
        module = DDP(
            module,
            device_ids=[device.index],
            find_unused_parameters=train_config["distributed_find_unused_params"],
        )
    # Data loop
    for epoch in range(1, train_config["finetune_epochs"] + 1):
        observers.on_fine_tuning_epoch_start(locals())
        #
        logger.info("Fine-Tune|Epoch %d", epoch)
        #
        if type(train_dataloader.sampler) is DistributedSampler:
            train_dataloader.sampler.set_epoch(epoch)
        #
        train_avg_loss, train_time = train_loop(
            train_dataloader,
            module,
            loss,
            optimizer,
            device,
            train_config["gradient_accumulation_steps"],
            train_config["clip_grad_norm"],
        )
        logger.info(
            "Fine-Tune|Train|Epoch %d|Time: %f s|Loss: %f",
            epoch,
            train_time,
            train_avg_loss,
        )
        #
        observers.on_fine_tuning_epoch_validation(locals())
        #
        valid_avg_loss, valid_time = valid_loop(valid_dataloader, module, loss, device)
        logger.info(
            "Fine-Tune|Valid|Epoch %d|Time: %f s|Loss: %f",
            epoch,
            valid_time,
            valid_avg_loss,
        )
        #
        logger.info(
            "Fine-Tune|Epoch %d|Time: %f s|Loss: %f",
            epoch,
            train_time + valid_time,
            train_avg_loss + valid_avg_loss,
        )
        #
        _scheduler_step(scheduler, train_avg_loss)
        #
        observers.on_fine_tuning_epoch_end(locals())
    # Save module checkpoint
    if not is_distributed() or rank == 0:
        if train_config["save_finetune_checkpoint"]:
            checkpoint_saver.save_checkpoint(
                checkpointer,
                train_config["epochs"] + train_config["finetune_epochs"],
                0,
                0,
                module,
                optimizer,
                scheduler,
            )
    # Unwrap module if distributed
    if type(module) is DDP:
        module = cast(nn.Module, module.module)
    # Apply optimizations
    module = Optimizations.apply(module, optimizations, "FINETUNE_END", locals())
    # Activate observer
    observers.on_fine_tuning_end(locals())
    # Return module
    return module


def _test(
    module: nn.Module,
    test_dataset: SpeechSeparationDatasetType,
    optimizations: list[Optimization],
    checkpointer: Checkpointer,
    checkpoint_name: Optional[str],
    observers: TrainingObservers,
    loss: loss.Loss,
    device: torch.device,
    seed: int,
    train_config: ReducedTrainingConfig,
) -> nn.Module:
    logger.info(
        "Using test metrics: %s",
        ", ".join(metric.__name__ for metric in train_config["test_metrics"]),
    )
    # Setup variables
    test_dataloader = _get_dataloader(test_dataset, device, seed, train_config)
    # If we didn't train we need to load a checkpoint
    if train_config["test_only"]:
        assert checkpoint_name is not None
        _load_test_checkpoint(checkpointer, checkpoint_name, module)
        logger.info("Using checkpoint for testing: %s", checkpoint_name)
    # Apply optimizations and begin data loop
    module = Optimizations.apply(module, optimizations, "TEST_START", locals())
    # Transfer the module to the device
    module = module.to(device)
    # Activate observer
    observers.on_testing_start(locals())
    # Data loop
    test_avg_loss, test_avg_metrics, test_time = test_loop(
        test_dataloader,
        module,
        loss,
        train_config["test_metrics"],
        device,
        train_config["calculate_test_metrics_improvement"],
    )
    # Log data
    metrics_str = ", ".join(["%f"] * len(test_avg_metrics))
    logger.info(
        f"Test|Time: %f s|Loss: %f|Metrics: {metrics_str}",
        test_time,
        test_avg_loss,
        *test_avg_metrics,
    )
    # Activate observer
    observers.on_testing_end(locals())
    # Return module
    return module


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
    loss = train_config["loss"]
    if train_config["convert_loss_to_permutation_invariant"] != "no":
        loss = PermutationInvariantMetric(
            loss,
            train_config["convert_loss_to_permutation_invariant"] == "greedy",
        )
    observers = TrainingObservers(train_config["observers"])
    module = model.get_module()
    # Log variables
    logger.info("Using id: %s", identifier)
    logger.info("Using seed: %d", seed)
    logger.info("Using loss: %s", train_config["loss"].__name__)
    # Activate observer
    observers.on_program_start(locals())
    # Load test dataset as it is always used
    test_dataset = dataset.get_test()
    # Train (if enabled)
    if not train_config["test_only"]:
        # Load train and valid dataset as they are shared
        train_dataset = dataset.get_train()
        valid_dataset = dataset.get_valid()
        # Run train
        module = _train(
            module,
            model,
            train_dataset,
            valid_dataset,
            test_dataset,
            optimizations,
            checkpointer,
            checkpoint_name,
            checkpoint_saver,
            observers,
            loss,
            device,
            seed,
            train_config,
        )
        # While optimizations require finetune repeat
        while Optimizations.requireFinetune(optimizations):
            module = _fine_tune(
                module,
                model,
                train_dataset,
                valid_dataset,
                test_dataset,
                optimizations,
                checkpointer,
                checkpoint_saver,
                observers,
                loss,
                device,
                seed,
                train_config,
            )
    # Run test
    module = _test(
        module,
        test_dataset,
        optimizations,
        checkpointer,
        checkpoint_name,
        observers,
        loss,
        device,
        seed,
        train_config,
    )
    # Activate observer
    observers.on_program_end(locals())
