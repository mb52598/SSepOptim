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
import torch
import fast_bss_eval.torch as fbss_torch

from ssepoptim.metrics.pit import fbss_pit_wrap
from ssepoptim.metrics.sdr import (
    scale_invariant_signal_to_distortion_ratio as _scale_invariant_signal_to_distortion_ratio,
)


@torch.compile
def signal_to_noise_ratio(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Calculate signal to noise ratio

    Parameters
    ----------
    prediction : torch.Tensor
        Prediction tensor of shape [batch, channel, time]
    target : torch.Tensor
        Target tensor of shape [batch, channel, time]

    Returns
    -------
    torch.Tensor
        SNR loss tensor of shape [batch, channel]
    """
    eps = torch.finfo(prediction.dtype).eps
    signal = target
    noise = prediction - signal
    snr = torch.sum(signal**2, dim=-1) / (torch.sum(noise**2, dim=-1) + eps)
    return 10 * torch.log10(snr + eps)


def scale_invariant_signal_to_noise_ratio(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Calculate scale invariant signal to noise ratio

    Parameters
    ----------
    prediction : torch.Tensor
        Prediction tensor of shape [batch, channel, time]
    target : torch.Tensor
        Target tensor of shape [batch, channel, time]

    Returns
    -------
    torch.Tensor
        SI-SNR loss tensor of shape [batch, channel]
    """
    return _scale_invariant_signal_to_distortion_ratio(
        prediction, target, zero_mean=True
    )


def scale_invariant_signal_to_distortion_ratio(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Calculate scale invariant signal to distortion ratio

    Parameters
    ----------
    prediction : torch.Tensor
        Prediction tensor of shape [batch, channel, time]
    target : torch.Tensor
        Target tensor of shape [batch, channel, time]

    Returns
    -------
    torch.Tensor
        SI-SDR loss tensor of shape [batch, channel]
    """
    return _scale_invariant_signal_to_distortion_ratio(
        prediction, target, zero_mean=False
    )


def signal_to_noise_ratio_pit(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return fbss_pit_wrap(signal_to_noise_ratio, prediction, target)


def scale_invariant_signal_to_noise_ratio_pit(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return fbss_pit_wrap(scale_invariant_signal_to_noise_ratio, prediction, target)


def scale_invariant_signal_to_distortion_ratio_pit(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return fbss_pit_wrap(scale_invariant_signal_to_distortion_ratio, prediction, target)


def signal_to_distortion_ratio_fbss_pit(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    try:
        result = fbss_torch.sdr(target, prediction)
    except:
        result = torch.tensor([torch.inf])
    return result
