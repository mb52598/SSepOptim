import logging
import os
from datetime import datetime
from typing import Literal, Optional

import ssepoptim.datasets as _
import ssepoptim.models as _
import ssepoptim.optimizations as _
from ssepoptim.base.checkpointing import CheckpointerConfig
from ssepoptim.base.configuration import BaseConfig, ConfigLoader
from ssepoptim.dataset import SpeechSeparationDatasetFactory
from ssepoptim.model import ModelFactory
from ssepoptim.optimization import OptimizationFactory
from ssepoptim.training_inference import TrainingInferenceConfig, train_test

logger = logging.getLogger(__name__)


class MainConfig(BaseConfig):
    tag: Optional[str]
    model: str
    dataset: str
    optimizations: list[str]
    logs_path: Optional[str]
    log_level: Literal[
        "CRITICAL", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"
    ]


def main(config_path: str):
    # Configurations
    loader = ConfigLoader(config_path)
    config = loader.get_config(MainConfig)
    train_infer_config = loader.get_config(TrainingInferenceConfig)
    ModelConfigClass = ModelFactory.get_config(config["model"])
    model_config = loader.get_config(ModelConfigClass)
    DatasetConfigClass = SpeechSeparationDatasetFactory.get_config(config["dataset"])
    dataset_config = loader.get_config(DatasetConfigClass)
    OptimizationConfigClasses = [
        OptimizationFactory.get_config(optimization)
        for optimization in config["optimizations"]
    ]
    optimization_configs = [
        loader.get_config(OpimizationConfigClass)
        for OpimizationConfigClass in OptimizationConfigClasses
    ]
    checkpointer_config = loader.get_config(CheckpointerConfig)
    # Logging
    logging.basicConfig(
        filename=(
            os.path.join(config["logs_path"], f"{datetime.now().isoformat()}.log")
            if config["logs_path"]
            else None
        ),
        filemode="x",
        format="[{levelname:^8s}] {asctime} {name:32s} {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
        level=getattr(logging, config["log_level"]),
    )
    # Print tag
    if config["tag"] is not None:
        logger.info(config["tag"])
    # Log configuration
    logger.info("Using model: %s", config["model"])
    logger.info("Using dataset: %s", config["dataset"])
    logger.info(
        "Using optimizations: %s",
        ", ".join(config["optimizations"]),
    )
    # Train and test
    train_test(
        config["model"],
        config["dataset"],
        config["optimizations"],
        model_config,
        dataset_config,
        optimization_configs,
        checkpointer_config,
        train_infer_config,
    )
