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

logger = logging.getLogger(__name__)


def train_test(
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
    checkpoint_name = None
    if train_config["load_last_checkpoint"] or train_config["test_only"]:
        checkpoint = search_and_load_checkpoint(
            checkpointer,
            model_name,
            model_config,
            dataset_config,
        )
        if checkpoint is None:
            raise RuntimeError("Unable to find any checkpoints")
        checkpoint_name, visible_metadata, hidden_metadata, _ = checkpoint
        identifier = visible_metadata[CheckpointerKeys.ID]
        seed = hidden_metadata[CheckpointerKeys.SEED]
        logger.info("Using checkpoint: %s", checkpoint_name)
    else:
        # Use id from config or create our own
        if train_config["id"] is None:
            identifier = str(uuid4())
        else:
            identifier = train_config["id"]
        # Use seed from config or create our own
        if train_config["seed"] is not None:
            seed = train_config["seed"]
        else:
            seed = random.randrange(sys.maxsize)
    # Start training
    if train_config["distributed_training"]:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        train_test_loop(
            identifier,
            seed,
            checkpoint_name,
            torch.device("cuda", device_id),
            model_name,
            dataset_name,
            optimization_names,
            model_config,
            dataset_config,
            optimization_configs,
            train_config,
        )
        dist.destroy_process_group()
    else:
        if train_config["device"] is not None:
            torch.set_default_device(train_config["device"])
        train_test_loop(
            identifier,
            seed,
            checkpoint_name,
            torch.get_default_device(),
            model_name,
            dataset_name,
            optimization_names,
            model_config,
            dataset_config,
            optimization_configs,
            train_config,
        )
