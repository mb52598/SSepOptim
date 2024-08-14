from typing import Callable

import torch

from ssepoptim.metric import (
    scale_invariant_signal_to_distortion_ratio,
    scale_invariant_signal_to_distortion_ratio_pit,
    scale_invariant_signal_to_noise_ratio,
    scale_invariant_signal_to_noise_ratio_pit,
    signal_to_noise_ratio,
)

Loss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def signal_to_noise_ratio_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return -signal_to_noise_ratio(prediction, target)


def scale_invariant_signal_to_noise_ratio_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return -scale_invariant_signal_to_noise_ratio(prediction, target)


def scale_invariant_signal_to_noise_ratio_pit_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return -scale_invariant_signal_to_noise_ratio_pit(prediction, target)


def scale_invariant_signal_to_distortion_ratio_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return -scale_invariant_signal_to_distortion_ratio(prediction, target)


def scale_invariant_signal_to_distortion_ratio_pit_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return -scale_invariant_signal_to_distortion_ratio_pit(prediction, target)
