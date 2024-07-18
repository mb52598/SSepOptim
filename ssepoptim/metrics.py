# [Open Source Tools & Data for Music Source Separation](https://source-separation.github.io/tutorial/index.html)
#
# E. Vincent, R. Gribonval and C. Fevotte,
# "Performance measurement in blind audio source separation,"
# in IEEE Transactions on Audio, Speech, and Language Processing,
# vol. 14, no. 4, pp. 1462-1469, July 2006, doi: 10.1109/TSA.2005.858005.
#
# Le Roux, J., Wisdom, S., Erdogan, H., & Hershey, J. R.
# (2019, May). SDR–half-baked or well done?.
# In ICASSP 2019-2019 IEEE International Conference on Acoustics,
# Speech and Signal Processing (ICASSP) (pp. 626-630). IEEE.
#
from typing import Callable

import torch

Metric = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def signal_to_noise_ratio(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Calculate signal to noise ratio

    Parameters
    ----------
    prediction : torch.Tensor
        Prediction tensor of shape (..., time)
    target : torch.Tensor
        Target tensor of shape (..., time)

    Returns
    -------
    torch.Tensor
        Differentiable tensor representing SNR loss
    """
    eps = torch.finfo(prediction.dtype).eps
    signal = target
    noise = target - prediction
    snr = (torch.sum(signal**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    return 10 * torch.log10(snr)


def scale_invariant_signal_to_noise_ratio(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Calculate scale invariant signal to noise ratio

    Parameters
    ----------
    prediction : torch.Tensor
        Prediction tensor of shape (..., time)
    target : torch.Tensor
        Target tensor of shape (..., time)

    Returns
    -------
    torch.Tensor
        Differentiable tensor representing SNR loss
    """
    raise NotImplementedError("U wot")


def scale_invariant_signal_to_distortion_ratio(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Calculate scale invariant signal to distortion ratio

    Parameters
    ----------
    prediction : torch.Tensor
        Prediction tensor of shape (..., time)
    target : torch.Tensor
        Target tensor of shape (..., time)

    Returns
    -------
    torch.Tensor
        Differentiable tensor representing SNR loss
    """
    eps = torch.finfo(prediction.dtype).eps

    optimal_scaling = (torch.sum(target * prediction, dim=-1, keepdim=True) + eps) / (
        torch.sum(target**2, dim=-1, keepdim=True) + eps
    )

    projection = optimal_scaling * target

    noise = prediction - projection

    ratio = (torch.sum(projection**2, dim=-1) + eps) / (
        torch.sum(noise**2, dim=-1) + eps
    )

    return 10 * torch.log10(ratio)
