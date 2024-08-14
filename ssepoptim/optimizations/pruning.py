from typing import Any, Callable, Literal, Type

import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.ao.pruning.sparsifier.weight_norm_sparsifier import WeightNormSparsifier

from ssepoptim.optimization import (
    Optimization,
    OptimizationConfig,
    OptimizationFactory,
    OptimizationStage,
)
from ssepoptim.utils.module_transforming import for_module
from ssepoptim.utils.type_checker import check_config_entries


class PruningOptimizationConfig(OptimizationConfig):
    method: Literal[
        "random-structured",
        "l2-structured",
        "l1-structured",
        "random-unstructured",
        "l1-unstructured",
    ]
    amount: float
    layers: list[Type[nn.Module]]


class PruningOptimization(Optimization):
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

    def _apply_pruning(self, layer: nn.Module, name: str):
        match self._config["method"]:
            case "random-structured":
                prune.random_structured(
                    layer, name, amount=self._config["amount"], dim=0
                )
            case "l2-structured":
                prune.ln_structured(
                    layer, name, amount=self._config["amount"], n=2, dim=0
                )
            case "l1-structured":
                prune.ln_structured(
                    layer, name, amount=self._config["amount"], n=1, dim=0
                )
            case "random-unstructured":
                prune.random_unstructured(layer, name, amount=self._config["amount"])
            case "l1-unstructured":
                prune.l1_unstructured(layer, name, amount=self._config["amount"])

    def _remove_pruning(self, layer: nn.Module, name: str):
        prune.remove(layer, name)

    def _apply_pruning_function(
        self, layer: nn.Module, function: Callable[[nn.Module, str], None]
    ):
        names = self._name_dict[type(layer)]
        for name in names:
            if not hasattr(layer, name):
                continue
            function(layer, name)

    def apply(
        self, module: nn.Module, stage: OptimizationStage, locals: dict[str, Any]
    ) -> nn.Module:
        match stage:
            case "FINETUNE_START":
                for layer_type in self._config["layers"]:
                    for_module(
                        module,
                        layer_type,
                        lambda x: self._apply_pruning_function(x, self._apply_pruning),
                    )
            case "FINETUNE_END":
                for layer_type in self._config["layers"]:
                    for_module(
                        module,
                        layer_type,
                        lambda x: self._apply_pruning_function(x, self._remove_pruning),
                    )
            case _:
                pass
        return module

    def requiresFinetune(self) -> bool:
        return True


class PruningOptimizationFactory(OptimizationFactory):
    @staticmethod
    def _get_config():
        return PruningOptimizationConfig

    @staticmethod
    def _get_object(config: OptimizationConfig):
        return PruningOptimization(
            check_config_entries(config, PruningOptimizationConfig)
        )
