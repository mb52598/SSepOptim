from itertools import permutations
from typing import Callable

import torch

from ssepoptim.metrics import (
    scale_invariant_signal_to_distortion_ratio,
    scale_invariant_signal_to_noise_ratio,
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


def scale_invariant_signal_to_distortion_ratio_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return -scale_invariant_signal_to_distortion_ratio(prediction, target)


def create_permutation_invariant_loss(loss: Loss) -> Loss:
    def pil(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Permutation invariant wrapper around provided loss

        Parameters
        ----------
        prediction : torch.Tensor
            Tensor of predictions of shape [batch, channel, time]
        target : torch.Tensor
            Tensor of targets of shape [batch, channel, time]

        Returns
        -------
        torch.Tensor
            Tensor of losses of shape [batch]
        """
        # TODO: Speed this up
        losses: list[torch.Tensor] = []
        channels = prediction.shape[1]
        for pred, tgt in zip(prediction, target):
            # Dim: [channel, time]
            pred = pred.unsqueeze(0).expand(channels, -1, -1)
            tgt = tgt.unsqueeze(1).expand(-1, channels, -1)
            # Dim: [channel, channel, time]
            loss_mat = loss(pred, tgt)
            # Dim: [channel, channel, time?]
            assert loss_mat.dim() in [2, 3]
            if loss_mat.dim() > 2:
                loss_mat = torch.mean(loss_mat, dim=-1)
            # Dim: [channel, channel]
            lowest_loss = None
            for p in permutations(range(channels)):
                candidate_loss = torch.mean(loss_mat[range(channels), p])
                if lowest_loss is None or candidate_loss < lowest_loss:
                    lowest_loss = candidate_loss
            # Dim: [1]
            assert lowest_loss is not None
            losses.append(lowest_loss)
        return torch.stack(losses)

    return pil
