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
    snr = (torch.sum(signal**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    return 10 * torch.log10(snr)


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
    raise NotImplementedError("U wot")


def _get_mask(source: torch.Tensor, source_lengths: torch.Tensor):
    """
    Arguments
    ---------
    source : torch.Tensor
        Shape [T, B, C]
    source_lengths : torch.Tensor
        Shape [B]

    Returns
    -------
    mask : torch.Tensor
        Shape [T, B, 1]

    Example
    -------
    >>> source = torch.randn(4, 3, 2)
    >>> source_lengths = torch.Tensor([2, 1, 4]).int()
    >>> mask = get_mask(source, source_lengths)
    >>> print(mask)
    tensor([[[1.],
             [1.],
             [1.]],
    <BLANKLINE>
            [[1.],
             [0.],
             [1.]],
    <BLANKLINE>
            [[0.],
             [0.],
             [1.]],
    <BLANKLINE>
            [[0.],
             [0.],
             [1.]]])
    """
    mask = source.new_ones(source.size()[:-1]).unsqueeze(-1).transpose(1, -2)
    B = source.size(-2)
    for i in range(B):
        mask[source_lengths[i] :, i] = 0
    return mask.transpose(-2, 1)


def _speechbrain_scale_invariant_signal_to_noise_ratio(
    estimate_source: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    """Calculate SI-SNR.

    Arguments
    ---------
    source: torch.Tensor
        Shape is [T, B, C],
        Where B is batch size, T is the length of the sources, C is the number of sources
        the ordering is made so that this loss is compatible with the class PitWrapper.
    estimate_source: torch.Tensor
        The estimated source, of shape [T, B, C]

    Returns
    -------
    The calculated SI-SNR.

    Example:
    ---------
    >>> import numpy as np
    >>> x = torch.Tensor([[1, 0], [123, 45], [34, 5], [2312, 421]])
    >>> xhat = x[:, (1, 0)]
    >>> x = x.unsqueeze(-1).repeat(1, 1, 2)
    >>> xhat = xhat.unsqueeze(1).repeat(1, 2, 1)
    >>> si_snr = -cal_si_snr(x, xhat)
    >>> print(si_snr)
    tensor([[[ 25.2142, 144.1789],
             [130.9283,  25.2142]]])
    """
    EPS = 1e-8
    assert source.size() == estimate_source.size()
    device = estimate_source.device.type

    source_lengths = torch.tensor(
        [estimate_source.shape[0]] * estimate_source.shape[-2], device=device
    )
    mask = _get_mask(source, source_lengths)
    estimate_source = estimate_source * mask

    num_samples = source_lengths.contiguous().reshape(1, -1, 1).float()  # [1, B, 1]
    mean_target = torch.sum(source, dim=0, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=0, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target = zero_mean_target * mask
    zero_mean_estimate = zero_mean_estimate * mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = zero_mean_target  # [T, B, C]
    s_estimate = zero_mean_estimate  # [T, B, C]
    # s_target = <s', s>s / ||s||^2
    dot = torch.sum(s_estimate * s_target, dim=0, keepdim=True)  # [1, B, C]
    s_target_energy = torch.sum(s_target**2, dim=0, keepdim=True) + EPS  # [1, B, C]
    proj = dot * s_target / s_target_energy  # [T, B, C]
    # e_noise = s' - s_target
    e_noise = s_estimate - proj  # [T, B, C]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    si_snr_beforelog = torch.sum(proj**2, dim=0) / (torch.sum(e_noise**2, dim=0) + EPS)
    si_snr = 10 * torch.log10(si_snr_beforelog + EPS)  # [B, C]

    return si_snr


def speechbrain_scale_invariant_signal_to_noise_ratio(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return _speechbrain_scale_invariant_signal_to_noise_ratio(
        prediction.permute(2, 0, 1), target.permute(2, 0, 1)
    )


def scale_invariant_signal_to_distortion_ratio(
    prediction: torch.Tensor, target: torch.Tensor
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
