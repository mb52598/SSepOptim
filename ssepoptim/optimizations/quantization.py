import logging
from typing import Any, Literal, Never, Optional, Type, cast

import torch
import torch.ao.quantization
import torch.ao.quantization as quantization
import torch.ao.quantization.quantize_fx as quantize_fx
import torch.ao.quantization.quantize_pt2e as quantize_pt2e
import torch.nn as nn
from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch.export import export as torch_export
from torch.utils.data import DataLoader

from ssepoptim.dataset import SpeechSeparationDataset
from ssepoptim.optimization import (
    Optimization,
    OptimizationConfig,
    OptimizationFactory,
    OptimizationStage,
)
from ssepoptim.training.base import TrainingConfig, collate_fn, sample_dataset
from ssepoptim.utils.context_timer import CtxTimer
from ssepoptim.utils.module_transforming import replace_module
from ssepoptim.utils.type_checker import check_config_entries

logger = logging.getLogger(__name__)


class QuantizationOptimizationConfig(OptimizationConfig):
    method: Literal["PTQ", "PTDQ", "QAT"]
    layers: Optional[list[Type[nn.Module]]]
    dtype: Literal["qint8", "float16"]
    implementation: Literal["Eager", "FX", "P2E-x86", "P2E-XNNPack"]


def _get_local_data(locals: dict[str, Any]):
    dataset = cast(SpeechSeparationDataset, locals["dataset"]).get_train()
    device = cast(torch.device, locals["device"])
    train_config = cast(TrainingConfig, locals["train_config"])
    return dataset, device, train_config


def _get_dataset_data(locals: dict[str, Any]):
    dataset, device, train_config = _get_local_data(locals)
    return (
        sample_dataset(dataset, batch_size=train_config["batch_size"])[0].to(device),
    )


def _calibrate_module(module: nn.Module, locals: dict[str, Any]):
    logger.info("Calibrating module on a single dataset run")
    dataset, _, train_config = _get_local_data(locals)
    device = torch.get_default_device()
    dataloader = DataLoader(
        dataset,
        batch_size=train_config["batch_size"],
        num_workers=train_config["num_workers"],
        collate_fn=collate_fn,
        generator=torch.Generator(device),
    )
    timer = CtxTimer()
    with torch.no_grad():
        mix: torch.Tensor
        for mix, _ in dataloader:
            module(mix.to(device))
    logger.info("Calibration done, time taken: %f", timer.total)


def _add_quant_to_module(module: nn.Module, qconfig: quantization.QConfig):
    new_module = nn.Sequential(
        quantization.QuantStub(), module, quantization.DeQuantStub()
    )
    setattr(new_module, "qconfig", qconfig)
    return new_module


def _get_quant_info(
    dtype_str: str,
    qscheme_type: Literal["per_tensor", "per_channel"],
):
    dtype: torch.dtype = getattr(torch, dtype_str)
    info = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
    if info.min == 0 or info.max == 0:
        qscheme = (
            torch.per_channel_affine
            if qscheme_type == "per_channel"
            else torch.per_tensor_affine
        )
        return dtype, qscheme, info.min, info.max
    best = min(abs(info.min), abs(info.max))
    if info.max < 0 or info.min > 0:
        raise NotImplementedError("Cannot handle this case")
    qscheme = (
        torch.per_channel_symmetric
        if qscheme_type == "per_channel"
        else torch.per_tensor_symmetric
    )
    return dtype, qscheme, -best, best


def _get_weight_activation_quant_info(
    activation_dtype_str: str,
    activation_qscheme_type: Literal["per_tensor", "per_channel"],
    weight_dtype_str: str,
    weight_qscheme_type: Literal["per_tensor", "per_channel"],
):
    activation_dtype, activation_qscheme, activation_min, activation_max = (
        _get_quant_info(activation_dtype_str, activation_qscheme_type)
    )
    weight_dtype, weight_qscheme, weight_min, weight_max = _get_quant_info(
        weight_dtype_str, weight_qscheme_type
    )
    return (
        (activation_dtype, activation_qscheme, activation_min, activation_max),
        (weight_dtype, weight_qscheme, weight_min, weight_max),
    )


def _get_int16_qconfig():
    int16_info = torch.iinfo(torch.int16)
    observer = quantization.default_observer.with_args(
        dtype=torch.int16,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
        quant_min=int16_info.min,
        quant_max=int16_info.max,
    )
    return quantization.QConfig(activation=observer, weight=observer)


class _PTQuantizationOptimization(Optimization):
    def __init__(self, config: QuantizationOptimizationConfig):
        self._config = config
        self._fine_tuned = False

    def apply(
        self, module: nn.Module, stage: OptimizationStage, locals: dict[str, Any]
    ) -> nn.Module:
        if stage != "FINETUNE_START":
            return module
        # Save state
        training = module.training
        # Prepare model
        module.eval()
        # Setup quantization
        data = _get_dataset_data(locals)
        layers = self._config["layers"] or []
        match self._config["implementation"]:
            case "Eager":
                match self._config["dtype"]:
                    case "qint8":
                        qconfig = quantization.default_qconfig
                    case "float16":
                        raise RuntimeError("Eager does not support PTQ with float16")
                for layer in layers:
                    replace_module(
                        module, layer, lambda x: _add_quant_to_module(x, qconfig)
                    )
                quantization.prepare(module, inplace=True)
                prepared_module = module
            case "FX":
                mapping = quantization.get_default_qconfig_mapping()
                prepared_module = quantize_fx.prepare_fx(
                    module,
                    mapping,
                    data,
                )
            case "P2E-x86" | "P2E-XNNPack":
                graph = torch_export(module, data)
                if self._config["implementation"] == "P2E-XNNPack":
                    quantizer = XNNPACKQuantizer().set_global(
                        get_symmetric_quantization_config()
                    )
                else:
                    quantizer = X86InductorQuantizer().set_global(
                        get_default_x86_inductor_quantization_config()
                    )
                prepared_module = quantize_pt2e.prepare_pt2e(graph, quantizer)
        # Calibrate the module
        _calibrate_module(prepared_module, locals)
        # Apply quantization
        match self._config["implementation"]:
            case "Eager":
                quantization.convert(prepared_module, inplace=True)
                quantized_module = prepared_module
            case "FX":
                quantized_module = quantize_fx.convert_fx(prepared_module)
            case "P2E-x86" | "P2E-XNNPack":
                quantized_module = quantize_pt2e.convert_pt2e(prepared_module)
        # Return state
        quantized_module.train(training)
        # Set flag
        self._fine_tuned = True
        # Return module
        return quantized_module

    def requiresFinetune(self) -> bool:
        return not self._fine_tuned


class _PTDQuantizationOptimization(Optimization):
    def __init__(self, config: QuantizationOptimizationConfig):
        self._config = config

    def apply(
        self, module: nn.Module, stage: OptimizationStage, locals: dict[str, Any]
    ) -> nn.Module:
        if stage != "FINETUNE_START":
            return module
        data = _get_dataset_data(locals)
        layers = self._config["layers"] or []
        match self._config["implementation"]:
            case "Eager":
                match self._config["dtype"]:
                    case "qint8":
                        qconfig = quantization.default_dynamic_qconfig
                    case "float16":
                        qconfig = quantization.float16_dynamic_qconfig
                qconfig_spec = {layer: qconfig for layer in layers}
                quantized_module = quantization.quantize_dynamic(module, qconfig_spec)
            case "FX":
                qconfig = quantization.default_dynamic_qconfig
                mapping = quantization.QConfigMapping().set_global(qconfig)
                prepared_module = quantize_fx.prepare_fx(module, mapping, data)
                quantized_module = quantize_fx.convert_fx(prepared_module)
            case "P2E-x86" | "P2E-XNNPack":
                raise RuntimeError(
                    "P2E does not support Post-Training Dynamic Quantization"
                )
        return quantized_module

    def requiresFinetune(self) -> bool:
        return False


class _QuantizationATOptimization(Optimization):
    def __init__(self, config: QuantizationOptimizationConfig):
        self._config = config

    def _p2e_not_supported(self) -> Never:
        raise RuntimeError("P2E does not support Quantization Aware Training")

    def _apply_finetune_start(
        self, module: nn.Module, locals: dict[str, Any]
    ) -> nn.Module:
        layers = self._config["layers"] or []
        match self._config["implementation"]:
            case "Eager":
                match self._config["dtype"]:
                    case "qint8":
                        qconfig = quantization.default_qat_qconfig
                    case "float16":
                        raise RuntimeError("Eager does not support QAT with float16")
                for layer in layers:
                    replace_module(
                        module, layer, lambda x: _add_quant_to_module(x, qconfig)
                    )
                quantization.prepare_qat(module, inplace=True)
                prepared_module = module
            case "FX":
                data = _get_dataset_data(locals)
                mapping = quantization.get_default_qat_qconfig_mapping()
                prepared_module = quantize_fx.prepare_qat_fx(
                    module,
                    mapping,
                    data,
                )
            case "P2E-x86" | "P2E-XNNPack":
                self._p2e_not_supported()
        return prepared_module

    def _apply_finetune_end(self, module: nn.Module) -> nn.Module:
        match self._config["implementation"]:
            case "Eager":
                quantization.convert(module, inplace=True)
                quantized_module = module
            case "FX":
                quantized_module = quantize_fx.convert_fx(module)
            case "P2E-x86" | "P2E-XNNPack":
                self._p2e_not_supported()
        return quantized_module

    def apply(
        self, module: nn.Module, stage: OptimizationStage, locals: dict[str, Any]
    ) -> nn.Module:
        match stage:
            case "TRAIN_START":
                quantized_module = self._apply_finetune_start(module, locals)
            case "TRAIN_END":
                quantized_module = self._apply_finetune_end(module)
            case _:
                quantized_module = module
        return quantized_module

    def requiresFinetune(self) -> bool:
        return False


class QuantizationOptimization(Optimization):
    def __init__(self, config: QuantizationOptimizationConfig):
        match config["method"]:
            case "PTQ":
                self._optimization = _PTQuantizationOptimization(config)
            case "PTDQ":
                self._optimization = _PTDQuantizationOptimization(config)
            case "QAT":
                self._optimization = _QuantizationATOptimization(config)

    def apply(
        self, module: nn.Module, stage: OptimizationStage, locals: dict[str, Any]
    ) -> nn.Module:
        return self._optimization.apply(module, stage, locals)

    def requiresFinetune(self) -> bool:
        return self._optimization.requiresFinetune()


class QuantizationOptimizationFactory(OptimizationFactory):
    @staticmethod
    def _get_config():
        return QuantizationOptimizationConfig

    @staticmethod
    def _get_object(config: OptimizationConfig):
        return QuantizationOptimization(
            check_config_entries(config, QuantizationOptimizationConfig)
        )
