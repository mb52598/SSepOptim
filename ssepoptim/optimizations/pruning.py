import logging
from typing import Any, Callable, Literal, Optional, Type

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.ao.pruning.sparsifier.weight_norm_sparsifier import WeightNormSparsifier

from ssepoptim.optimization import (
    Optimization,
    OptimizationConfig,
    OptimizationFactory,
    OptimizationStage,
)
from ssepoptim.optimizations.helpers.module_run import calculate_module_gradients
from ssepoptim.optimizations.pruning_impl.weight_change_pruning import (
    weight_change_structured,
    weight_change_unstructured,
)
from ssepoptim.utils.module_transforming import for_module
from ssepoptim.utils.type_checker import check_config_entries

logger = logging.getLogger(__name__)


class PruningOptimizationConfig(OptimizationConfig):
    method: Literal[
        "random-structured",
        "l2-structured",
        "l1-structured",
        "weight-change-structured",
        "random-unstructured",
        "l1-unstructured",
        "weight-change-unstructured",
    ]
    amount: float
    layers: list[Type[nn.Module]]
    num_iters: int
    model_validation_deterioration_delta: Optional[float]
    dataset_percentage: Optional[float]


def _get_pruning_masks(layer: nn.Module, name: str):
    masks: list[torch.Tensor] = []
    for hook in layer._forward_pre_hooks.values():
        if isinstance(hook, prune.BasePruningMethod) and hook._tensor_name == name:
            mask = getattr(layer, hook._tensor_name + "_mask")
            masks.append(mask)
    return masks


def _apply_pruning_masks_and_make_permanent(
    layer: nn.Module, name: str, masks: list[torch.Tensor]
):
    for hook in layer._forward_pre_hooks.values():
        if isinstance(hook, prune.BasePruningMethod) and hook._tensor_name == name:
            setattr(layer, hook._tensor_name + "_mask", masks.pop(0))
    prune.remove(layer, name)


class BasePruningOptimization(Optimization):
    def __init__(self, config: PruningOptimizationConfig):
        self._config = config
        self._name_dict = {
            nn.Module: [],
            nn.Linear: ["weight"],
            nn.Conv1d: ["weight"],
            nn.Conv2d: ["weight"],
            nn.Conv3d: ["weight"],
            nn.ConvTranspose1d: ["weight"],
            nn.ConvTranspose2d: ["weight"],
            nn.ConvTranspose3d: ["weight"],
            nn.MultiheadAttention: [
                "in_proj_weight",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
            ],
        }
        self._iter = 0
        self._masks: list[list[torch.Tensor]] = []
        self._prev_loss = float("inf")
        self._finetune_done = False

    def _apply_pruning(self, layer: nn.Module, name: str):
        # If we are watching for deterioration we need to save previous masks
        if self._config["model_validation_deterioration_delta"] is not None:
            # If its the first iteration we have no masks to save
            if self._iter != 0:
                self._masks.append(_get_pruning_masks(layer, name))
        # Apply pruning method
        match self._config["method"]:
            case "random-structured":
                prune.random_structured(
                    layer, name, amount=self._config["amount"], dim=-1
                )
            case "l2-structured":
                prune.ln_structured(
                    layer, name, amount=self._config["amount"], n=2, dim=-1
                )
            case "l1-structured":
                prune.ln_structured(
                    layer, name, amount=self._config["amount"], n=1, dim=-1
                )
            case "weight-change-structured":
                weight_change_structured(
                    layer, name, amount=self._config["amount"], dim=-1
                )
            case "random-unstructured":
                prune.random_unstructured(layer, name, amount=self._config["amount"])
            case "l1-unstructured":
                prune.l1_unstructured(layer, name, amount=self._config["amount"])
            case "weight-change-unstructured":
                weight_change_unstructured(layer, name, amount=self._config["amount"])

    def _make_pruning_permanent(self, layer: nn.Module, name: str):
        # If we are watching for deterioration we need to set saved masks
        if (
            self._config["model_validation_deterioration_delta"] is not None
            and len(self._masks) > 0
        ):
            _apply_pruning_masks_and_make_permanent(layer, name, self._masks.pop(0))
        # Otherwise just make pruning permanent
        else:
            prune.remove(layer, name)

    def _run_layer_function(
        self, layer: nn.Module, function: Callable[[nn.Module, str], None]
    ):
        names = self._name_dict[type(layer)]
        for name in names:
            if not hasattr(layer, name):
                logging.warn(
                    'Attempting to prune layer "%s" with parameter "%s" which is missing',
                    layer.__class__.__name__,
                    name,
                )
                continue
            function(layer, name)

    def _run_module_function(
        self, module: nn.Module, function: Callable[[nn.Module, str], None]
    ):
        for layer_type in self._config["layers"]:
            for_module(
                module,
                layer_type,
                lambda x: self._run_layer_function(x, function),
            )

    def _finetune_start(self, module: nn.Module, locals: dict[str, Any]):
        # We need to clear masks if they were set in the previous iteration
        self._masks.clear()
        # Apply the pruning method for all the configured layers
        self._run_module_function(module, self._apply_pruning)

    def _finetune_end(self, module: nn.Module, locals: dict[str, Any]):
        # Check for model deterioration
        if self._config["model_validation_deterioration_delta"] is not None:
            valid_avg_loss: float = locals["valid_avg_loss"]
            # If the change in loss is bigger than threshold than we are done
            if (
                valid_avg_loss - self._prev_loss
                > self._config["model_validation_deterioration_delta"]
            ):
                self._finetune_done = True
            # Save the new loss
            self._prev_loss = valid_avg_loss
        # Check for number of iterations
        if self._iter == self._config["num_iters"]:
            # We clear masks as there isn't sufficient deterioration to reset the masks
            # We know this, because we reached the iteration number condition
            #                           instead of the deterioration condition
            self._masks.clear()
            self._finetune_done = True
        # If we are done make pruning permanent
        if not self._finetune_done:
            logger.info("Doing another iteration of fine-tuning")
            return
        # Make pruning permanent
        self._run_module_function(module, self._make_pruning_permanent)

    def apply(
        self, module: nn.Module, stage: OptimizationStage, locals: dict[str, Any]
    ) -> nn.Module:
        match stage:
            case "FINETUNE_START":
                self._finetune_start(module, locals)
                self._iter += 1
            case "FINETUNE_END":
                self._finetune_end(module, locals)
            case _:
                pass
        return module

    def requiresFinetune(self) -> bool:
        return not self._finetune_done


class WeightChangePruningOptimization(BasePruningOptimization):
    def __init__(self, config: PruningOptimizationConfig):
        super().__init__(config)
        self._one_more_iter = False

    def _finetune_start(self, module: nn.Module, locals: dict[str, Any]):
        # We are doing one more iteration to fix pruning mistake so skip pruning
        # In the base class this is unnecessary as we undo the pruning but here
        #               we cannot undo the pruning so we finetune one more time
        if self._one_more_iter:
            return
        # Save state
        training = module.training
        # Switch to training
        module.train()
        # Calculate gradients
        logger.info("Calculating module gradients on a single dataset run")
        time_taken = calculate_module_gradients(
            module, locals, self._config["dataset_percentage"] or 1.0
        )
        logger.info("Gradient calculation done, time taken: %f", time_taken)
        # Apply the pruning method for all the configured layers
        self._run_module_function(module, self._apply_pruning)
        # We clear masks as we cannot undo weight change pruning
        self._masks.clear()
        # We need to return the state of the model as we changed it at the start
        module.train(training)

    def _finetune_end(self, module: nn.Module, locals: dict[str, Any]):
        # If we are doing one more iteration of fine-tuning, skip
        if self._one_more_iter:
            self._finetune_done = True
            return
        # Make pruning permanent
        self._run_module_function(module, self._make_pruning_permanent)
        # Check for model deterioration
        if self._config["model_validation_deterioration_delta"] is not None:
            valid_avg_loss: float = locals["valid_avg_loss"]
            # If the change in loss is bigger than threshold than we are done
            if (
                valid_avg_loss - self._prev_loss
                > self._config["model_validation_deterioration_delta"]
            ):
                self._one_more_iter = True
            # Save the new loss
            self._prev_loss = valid_avg_loss
        # Check for number of iterations
        if self._iter == self._config["num_iters"]:
            self._one_more_iter = True
        # Log message
        if self._one_more_iter:
            logger.info("Doing last iteration of fine-tuning")
        else:
            logger.info("Doing another iteration of fine-tuning")


def get_pruning_object(config: PruningOptimizationConfig):
    if config["method"].startswith("weight-change"):
        return WeightChangePruningOptimization(config)
    else:
        return BasePruningOptimization(config)


class PruningOptimizationFactory(OptimizationFactory):
    @staticmethod
    def _get_config():
        return PruningOptimizationConfig

    @staticmethod
    def _get_object(config: OptimizationConfig):
        return get_pruning_object(
            check_config_entries(config, PruningOptimizationConfig)
        )
