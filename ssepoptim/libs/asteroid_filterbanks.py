# This file is an extraction of needed modules from asteroid_filterbanks with added type annotations and additional error handling
# Original repo: https://github.com/asteroid-team/asteroid-filterbanks
from typing import Optional

import torch
import torch.nn as nn


class _EncDec(nn.Module):
    def __init__(
        self,
        n_filters: int,
        kernel_size: int,
        stride: Optional[int] = None,
    ):
        super().__init__()

        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride if stride else self.kernel_size // 2
        self.n_feats_out = n_filters


class Decoder(_EncDec):
    """Decoder class.

    Args:
        padding (int): Zero-padding added to both sides of the input.
        output_padding (int): Additional size added to one side of the
            output shape.
    """

    def __init__(
        self,
        n_filters: int,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
        output_padding: int = 0,
    ):
        super().__init__(n_filters, kernel_size, stride)

        self.padding = padding
        self.output_padding = output_padding

        self.conv_layer = nn.ConvTranspose1d(
            self.n_filters,
            1,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        nn.init.xavier_normal_(self.conv_layer.weight)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        view_as = (-1,) + spec.shape[-2:]
        out = self.conv_layer(spec.reshape(view_as))
        wav = out.view(spec.shape[:-2] + (-1,))

        return wav


class Encoder(_EncDec):
    r"""Encoder class.

    Args:
        filterbank (:class:`Filterbank`): The filterbank to use
            as an encoder.
        padding (int): Zero-padding added to both sides of the input.

    """

    def __init__(
        self,
        n_filters: int,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
    ):
        super().__init__(n_filters, kernel_size, stride)

        self.padding = padding

        self.conv_layer = nn.Conv1d(
            1,
            self.n_filters,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        nn.init.xavier_normal_(self.conv_layer.weight)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.conv_layer(waveform)
        return spec


def make_enc_dec(
    n_filters: int,
    kernel_size: int,
    stride: Optional[int] = None,
    padding: int = 0,
    output_padding: int = 0,
):
    enc = Encoder(n_filters, kernel_size, stride, padding)
    dec = Decoder(n_filters, kernel_size, stride, padding, output_padding)
    return enc, dec
