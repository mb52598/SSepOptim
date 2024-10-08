import torch
import torch.nn as nn
import torch.optim as optim

from ssepoptim.libs.speechbrain import (
    Decoder,
    Dual_Path_Model,
    Encoder,
    SBTransformerBlock,
)
from ssepoptim.model import Model, ModelConfig, ModelFactory
from ssepoptim.utils.type_checker import check_config_entries


class SepformerConfig(ModelConfig):
    N_encoder_out: int
    out_channels: int
    kernel_size: int
    kernel_stride: int
    num_spks: int


def get_encoder(config: SepformerConfig):
    return Encoder(
        kernel_size=config["kernel_size"], out_channels=config["N_encoder_out"]
    )


def get_decoder(config: SepformerConfig):
    return Decoder(
        in_channels=config["N_encoder_out"],
        out_channels=1,
        kernel_size=config["kernel_size"],
        stride=config["kernel_stride"],
        bias=False,
    )


def get_masknet(config: SepformerConfig):
    intra_model, inter_model = [
        SBTransformerBlock(
            num_layers=8,
            d_model=config["out_channels"],
            nhead=8,
            d_ffn=1024,
            dropout=0,
            use_positional_encoding=True,
            norm_before=True,
        )
        for _ in range(2)
    ]
    return Dual_Path_Model(
        in_channels=config["N_encoder_out"],
        out_channels=config["out_channels"],
        intra_model=intra_model,
        inter_model=inter_model,
        num_layers=2,
        norm="ln",
        K=250,
        num_spks=config["num_spks"],
        skip_around_intra=True,
        linear_layer_after_inter_intra=False,
    )


class SepformerModule(nn.Module):
    def __init__(self, config: SepformerConfig):
        super().__init__()

        self._config = config
        self._encoder = get_encoder(config)
        self._masknet = get_masknet(config)
        self._decoder = get_decoder(config)

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        mix = mix.squeeze(-2)
        encoded_mix = self._encoder(mix)
        est_mask = self._masknet(encoded_mix)
        encoded_mix_stack = torch.stack([encoded_mix] * self._config["num_spks"])
        encoded_separated_stack = encoded_mix_stack * est_mask
        return torch.cat(
            [
                self._decoder(encoded_separated_stack[i]).unsqueeze(-1)
                for i in range(self._config["num_spks"])
            ]
        )


class Sepformer(Model):
    def __init__(self, config: SepformerConfig):
        self._config = config

    def get_module(self) -> nn.Module:
        return SepformerModule(self._config)

    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=0.00015, weight_decay=0)

    def get_scheduler(
        self, optimizer: optim.Optimizer
    ) -> optim.lr_scheduler.LRScheduler:
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)


class SepformerFactory(ModelFactory):
    @staticmethod
    def _get_config():
        return SepformerConfig

    @staticmethod
    def _get_object(config: ModelConfig):
        return Sepformer(check_config_entries(config, SepformerConfig))
