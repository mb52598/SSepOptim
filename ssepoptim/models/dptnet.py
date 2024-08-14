from typing import Optional, Type

import torch.nn as nn
import torch.optim as optim

from ssepoptim.libs.asteroid import BaseEncoderMaskerDecoder, DPTransformer
from ssepoptim.libs.asteroid_filterbanks import Filterbank, make_enc_dec
from ssepoptim.model import Model, ModelConfig, ModelFactory
from ssepoptim.utils.type_checker import check_config_entries


class DPTNetConfig(ModelConfig):
    n_src: int
    n_heads: int
    ff_hid: int
    chunk_size: int
    hop_size: Optional[int]
    n_repeats: int
    norm_type: Type[nn.Module]
    ff_activation: Type[nn.Module]
    encoder_activation: Type[nn.Module]
    mask_act: Type[nn.Module]
    bidirectional: bool
    dropout: int
    fb_class: Type[Filterbank]
    kernel_size: int
    n_filters: int
    stride: int
    sample_rate: int


class DPTNetModule(BaseEncoderMaskerDecoder):
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

    def __init__(self, config: DPTNetConfig):
        encoder, decoder = make_enc_dec(
            config["fb_class"],
            n_filters=config["n_filters"],
            kernel_size=config["kernel_size"],
            stride=config["stride"],
            sample_rate=config["sample_rate"],
        )
        n_feats = encoder.n_feats_out

        # Update in_chan
        masker = DPTransformer(
            n_feats,
            config["n_src"],
            n_heads=config["n_heads"],
            ff_hid=config["ff_hid"],
            ff_activation=config["ff_activation"],
            chunk_size=config["chunk_size"],
            hop_size=config["hop_size"],
            n_repeats=config["n_repeats"],
            norm_type=config["norm_type"],
            mask_act=config["mask_act"],
            bidirectional=config["bidirectional"],
            dropout=config["dropout"],
        )
        super().__init__(
            encoder, masker, decoder, encoder_activation=config["encoder_activation"]
        )


class DPTNet(Model):
    def __init__(self, config: DPTNetConfig):
        self._config = config

    def get_module(self) -> nn.Module:
        return DPTNetModule(self._config)

    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        return optim.Adam(model.parameters())

    def get_scheduler(
        self, optimizer: optim.Optimizer
    ) -> optim.lr_scheduler.LRScheduler:
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)


class DPTNetFactory(ModelFactory):
    @staticmethod
    def _get_config():
        return DPTNetConfig

    @staticmethod
    def _get_object(config: ModelConfig):
        return DPTNet(check_config_entries(config, DPTNetConfig))
