from itertools import cycle, islice, permutations

import torch
import torch.nn as nn
from fast_bss_eval.torch.hungarian import linear_sum_assignment

from ssepoptim.metrics.base import Metric


class PermutationInvariantMetric(nn.Module):
    def __init__(self, metric: Metric, greedy: bool = False):
        super().__init__()

        self._metric = metric
        self._greedy = greedy

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Permutation invariant wrapper around provided metric

        Parameters
        ----------
        prediction : torch.Tensor
            Tensor of predictions of shape [batch, channel, time]
        target : torch.Tensor
            Tensor of targets of shape [batch, channel, time]

        Returns
        -------
        torch.Tensor
            Tensor of metrics of shape [batch]
        """
        # TODO: Speed this up
        # TODO: Add Hungarian algorithm
        metrics: list[torch.Tensor] = []
        channels = prediction.shape[1]
        for pred, tgt in zip(prediction, target):
            # Dim: [channel, time]
            pred = pred.unsqueeze(1).expand(-1, channels, -1)
            tgt = tgt.unsqueeze(0).expand(channels, -1, -1)
            # Dim: [channel, channel, time]
            cost_mat = self._metric(pred, tgt)
            # Dim: [channel, channel, time?]
            assert cost_mat.dim() in [2, 3]
            if cost_mat.dim() > 2:
                cost_mat = torch.mean(cost_mat, dim=-1)
            # Dim: [channel, channel]
            lowest_metric = None
            if self._greedy:
                # Complexity: channel * (((channel + 1) * channel) / 2) ~= channel ** 3
                channels_left_cycle = cycle(range(channels))
                for _ in range(channels):
                    channels_left = list(range(channels))
                    indexes: list[int] = []
                    row_indexes = list(islice(channels_left_cycle, 1, channels + 1))
                    for row_index in row_indexes:
                        index = int(cost_mat[row_index, channels_left].argmin().item())
                        indexes.append(channels_left[index])
                        del channels_left[index]
                    candidate_metric = torch.mean(cost_mat[row_indexes, indexes])
                    if lowest_metric is None or candidate_metric < lowest_metric:
                        lowest_metric = candidate_metric
            else:
                # Complexity: factorial(channel)
                for p in permutations(range(channels)):
                    candidate_metric = torch.mean(cost_mat[range(channels), p])
                    if lowest_metric is None or candidate_metric < lowest_metric:
                        lowest_metric = candidate_metric
            # Dim: [1]
            assert lowest_metric is not None
            metrics.append(lowest_metric)
        return torch.stack(metrics)


def fbss_pit_wrap(
    metric: Metric,
    prediction: torch.Tensor,
    target: torch.Tensor,
    maximize: bool = True,
) -> torch.Tensor:
    channels = prediction.shape[1]
    values: list[torch.Tensor] = []
    for pred, tgt in zip(prediction, target):
        pred = pred.unsqueeze(1).expand(-1, channels, -1)
        tgt = tgt.unsqueeze(0).expand(channels, -1, -1)
        cost_mat = metric(pred, tgt)
        assert cost_mat.dim() in [2, 3]
        if cost_mat.dim() > 2:
            cost_mat = torch.mean(cost_mat, dim=-1)
        rows, cols = linear_sum_assignment(-cost_mat if maximize else cost_mat)
        values.append(torch.mean(cost_mat[rows, cols]))
    return torch.stack(values)
