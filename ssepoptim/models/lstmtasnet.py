from copy import deepcopy
from typing import Optional, Type

import torch
import torch.nn as nn
import torch.optim as optim

from ssepoptim.libs.asteroid import BaseEncoderMaskerDecoder, LSTMMasker
from ssepoptim.libs.asteroid_filterbanks import Encoder, make_enc_dec
from ssepoptim.model import Model, ModelConfig, ModelFactory
from ssepoptim.utils.type_checker import check_config_entries


class LSTMTasNetConfig(ModelConfig):
    n_src: int
    out_chan: Optional[int]
    hid_size: int
    mask_act: Type[nn.Module]
    bidirectional: bool
    rnn_type: Type[nn.Module]
    n_layers: int
    dropout: float
    encoder_activation: Type[nn.Module]
    n_filters: int
    kernel_size: int
    stride: int


class _GatedEncoder(nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__()

        # Gated encoder.
        self.encoder_relu = encoder
        self.encoder_sig = deepcopy(encoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        relu_out = torch.relu(self.encoder_relu(x))
        sig_out = torch.sigmoid(self.encoder_sig(x))
        return sig_out * relu_out


class LSTMTasNetModule(BaseEncoderMaskerDecoder):
    """DPTNet separation model, as described in [1].

    Args:
        n_src (int): Number of masks to estimate.
        out_chan  (int or None): Number of bins in the estimated masks.
            Defaults to `in_chan`.
        bn_chan (int): Number of channels after the bottleneck.
            Defaults to 128.
        hid_size (int): Number of neurons in the RNNs cell state.
            Defaults to 128.
        chunk_size (int): window size of overlap and add processing.
            Defaults to 100.
        hop_size (int or None): hop size (stride) of overlap and add processing.
            Default to `chunk_size // 2` (50% overlap).
        n_repeats (int): Number of repeats. Defaults to 6.
        norm_type (str, optional): Type of normalization to use. To choose from

            - ``'gLN'``: global Layernorm
            - ``'cLN'``: channelwise Layernorm
        mask_act (str, optional): Which non-linear function to generate mask.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        rnn_type (str, optional): Type of RNN used. Choose between ``'RNN'``,
            ``'LSTM'`` and ``'GRU'``.
        num_layers (int, optional): Number of layers in each RNN.
        dropout (float, optional): Dropout ratio, must be in [0,1].
        in_chan (int, optional): Number of input channels, should be equal to
            n_filters.
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``].
        n_filters (int): Number of filters / Input dimension of the masker net.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        sample_rate (float): Sampling rate of the model.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.

    References
        - [1]: Jingjing Chen et al. "Dual-Path Transformer Network: Direct
          Context-Aware Modeling for End-to-End Monaural Speech Separation"
          Interspeech 2020.
    """

    def __init__(self, config: LSTMTasNetConfig):
        encoder, decoder = make_enc_dec(
            n_filters=config["n_filters"],
            kernel_size=config["kernel_size"],
            stride=config["stride"],
        )
        n_feats = encoder.n_feats_out

        # Real gated encoder
        encoder = _GatedEncoder(encoder)

        # Masker
        assert config["rnn_type"] in [nn.RNN, nn.LSTM, nn.GRU]
        masker = LSTMMasker(
            n_feats,
            config["n_src"],
            out_chan=config["out_chan"],
            hid_size=config["hid_size"],
            mask_act=config["mask_act"],
            bidirectional=config["bidirectional"],
            rnn_type=config["rnn_type"],
            n_layers=config["n_layers"],
            dropout=config["dropout"],
        )
        super().__init__(encoder, masker, decoder, config["encoder_activation"])


class LSTMTasNet(Model):
    def __init__(self, config: LSTMTasNetConfig):
        self._config = config

    def get_module(self) -> nn.Module:
        return LSTMTasNetModule(self._config)

    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    def get_scheduler(
        self, optimizer: optim.Optimizer
    ) -> optim.lr_scheduler.LRScheduler:
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)


class LSTMTasNetFactory(ModelFactory):
    @staticmethod
    def _get_config():
        return LSTMTasNetConfig

    @staticmethod
    def _get_object(config: ModelConfig):
        return LSTMTasNet(check_config_entries(config, LSTMTasNetConfig))
