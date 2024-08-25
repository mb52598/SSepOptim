import logging
import random
import sys
from uuid import uuid4

import torch
import torch.distributed as dist

from ssepoptim.base.checkpointing import Checkpointer
from ssepoptim.dataset import SpeechSeparationDatasetConfig
from ssepoptim.model import ModelConfig
from ssepoptim.optimization import OptimizationConfig
from ssepoptim.training.base import (
    CheckpointerKeys,
    TrainingConfig,
    search_and_load_checkpoint,
)
from ssepoptim.training.training_loop import train_test as train_test_loop
from ssepoptim.utils.distributed import get_local_rank, is_distributed

logger = logging.getLogger(__name__)


def _get_default_values(train_config: TrainingConfig):
    # Use id from config or create our own
    if train_config["id"] is None:
        identifier = str(uuid4())
    else:
        identifier = train_config["id"]
    # Use seed from config or create our own
    if train_config["seed"] is not None:
        seed = train_config["seed"]
    else:
        # Check for distributed
        if is_distributed():
            raise RuntimeError("Seed needs to be set for distributed training")
        # Generate seed
        seed = random.randrange(sys.maxsize)
    return identifier, seed


def _attemp_search_and_load_checkpoint(
    checkpointer: Checkpointer,
    model_name: str,
    model_config: ModelConfig,
    dataset_config: SpeechSeparationDatasetConfig,
    raise_on_failure: bool,
):
    checkpoint = search_and_load_checkpoint(
        checkpointer,
        model_name,
        model_config,
        dataset_config,
    )
    if checkpoint is None:
        if raise_on_failure:
            raise RuntimeError("Unable to find any checkpoints")
        return None
    checkpoint_name, visible_metadata, hidden_metadata, _ = checkpoint
    identifier = visible_metadata[CheckpointerKeys.ID]
    seed = hidden_metadata[CheckpointerKeys.SEED]
    return checkpoint_name, identifier, seed


def train_test(
    task_index: int,
    total_tasks: int,
    model_name: str,
    dataset_name: str,
    optimization_names: list[str],
    model_config: ModelConfig,
    dataset_config: SpeechSeparationDatasetConfig,
    optimization_configs: list[OptimizationConfig],
    train_config: TrainingConfig,
):
    # Load last checkpoint, setup id and seed
    checkpointer = Checkpointer(train_config["checkpoints_path"])
    checkpoint_data = None
    if train_config["load_last_checkpoint"] or train_config["test_only"]:
        checkpoint_data = _attemp_search_and_load_checkpoint(
            checkpointer,
            model_name,
            model_config,
            dataset_config,
            bool(train_config["test_only"]),
        )
    if checkpoint_data is not None:
        checkpoint_name, identifier, seed = checkpoint_data
        logger.info("Using checkpoint: %s", checkpoint_name)
    else:
        checkpoint_name = None
        identifier, seed = _get_default_values(train_config)
    # Start training
    if is_distributed():
        # Only initialize process group on first task
        if task_index == 0:
            dist.init_process_group(backend="nccl")
        # Log we are using distributed
        logger.info("Using distributed training")
        # Setup default device
        device_id = get_local_rank()
        torch.cuda.set_device(device_id)
        device = torch.device("cuda", device_id)
        torch.set_default_device(device)
        train_test_loop(
            identifier,
            seed,
            checkpoint_name,
            device,
            model_name,
            dataset_name,
            optimization_names,
            model_config,
            dataset_config,
            optimization_configs,
            train_config,
        )
        # Only delete the process group on the last task
        if task_index == total_tasks - 1:
            dist.destroy_process_group()
    else:
        if train_config["device"] is not None:
            device = torch.device(train_config["device"])
        else:
            device = torch.get_default_device()
        torch.set_default_device(device)
        train_test_loop(
            identifier,
            seed,
            checkpoint_name,
            device,
            model_name,
            dataset_name,
            optimization_names,
            model_config,
            dataset_config,
            optimization_configs,
            train_config,
        )
