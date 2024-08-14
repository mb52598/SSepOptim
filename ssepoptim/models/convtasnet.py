import warnings
from typing import Optional, Type

import torch.nn as nn
import torch.optim as optim

from ssepoptim.libs.asteroid import BaseEncoderMaskerDecoder, ChanLN, CumLN, TDConvNet
from ssepoptim.libs.asteroid_filterbanks import Filterbank, make_enc_dec
from ssepoptim.model import Model, ModelConfig, ModelFactory
from ssepoptim.utils.type_checker import check_config_entries


class ConvTasNetConfig(ModelConfig):
    n_src: int
    out_chan: Optional[int]
    n_blocks: int
    n_repeats: int
    bn_chan: int
    hid_chan: int
    skip_chan: int
    conv_kernel_size: int
    norm_type: Type[nn.Module]
    mask_act: Type[nn.Module]
    causal: bool
    fb_class: Type[Filterbank]
    n_filters: int
    kernel_size: int
    stride: int
    encoder_activation: Type[nn.Module]
    sample_rate: float


class ConvTasNetModule(BaseEncoderMaskerDecoder):
    """ConvTasNet separation model, as described in [1].

    Args:
        n_src (int): Number of sources in the input mixtures.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 3.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        skip_chan (int, optional): Number of channels in the skip connections.
            If 0 or None, TDConvNet won't have any skip connections and the
            masks will be computed from the residual output.
            Corresponds to the ConvTasnet architecture in v1 or the paper.
        conv_kernel_size (int, optional): Kernel size in convolutional blocks.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.
        mask_act (str, optional): Which non-linear function to generate mask.
        in_chan (int, optional): Number of input channels, should be equal to
            n_filters.
        causal (bool, optional) : Whether or not the convolutions are causal.
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
        - [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
          for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
          https://arxiv.org/abs/1809.07454
    """

    def __init__(self, config: ConvTasNetConfig):
        encoder, decoder = make_enc_dec(
            config["fb_class"],
            n_filters=config["n_filters"],
            kernel_size=config["kernel_size"],
            stride=config["stride"],
            sample_rate=config["sample_rate"],
        )
        n_feats = encoder.n_feats_out

        norm_type = config["norm_type"]
        if config["causal"] and config["norm_type"] not in [CumLN, ChanLN]:
            norm_type = CumLN
            warnings.warn(
                "In causal configuration cumulative layer normalization (cgLN)"
                "or channel-wise layer normalization (chanLN)  "
                f"must be used. Changing {norm_type} to cLN"
            )

        # Update in_chan
        masker = TDConvNet(
            n_feats,
            config["n_src"],
            out_chan=config["out_chan"],
            n_blocks=config["n_blocks"],
            n_repeats=config["n_repeats"],
            bn_chan=config["bn_chan"],
            hid_chan=config["hid_chan"],
            skip_chan=config["skip_chan"],
            conv_kernel_size=config["conv_kernel_size"],
            norm_type=norm_type,
            mask_act=config["mask_act"],
            causal=config["causal"],
        )
        super().__init__(encoder, masker, decoder, config["encoder_activation"])


class ConvTasNet(Model):
    def __init__(self, config: ConvTasNetConfig):
        self._config = config

    def get_module(self) -> nn.Module:
        return ConvTasNetModule(self._config)

    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    def get_scheduler(
        self, optimizer: optim.Optimizer
    ) -> optim.lr_scheduler.LRScheduler:
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)


class ConvTasNetFactory(ModelFactory):
    @staticmethod
    def _get_config():
        return ConvTasNetConfig

    @staticmethod
    def _get_object(config: ModelConfig):
        return ConvTasNet(check_config_entries(config, ConvTasNetConfig))
