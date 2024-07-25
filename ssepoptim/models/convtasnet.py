from torchaudio.models import ConvTasNet as ConvTasNetTorch

from ssepoptim.model import ModelConfig, ModelFactory
from ssepoptim.utils.type_checker import check_config_entries


class ConvTasNetConfig(ModelConfig):
    num_sources: int
    # encoder/decoder parameters
    enc_kernel_size: int
    enc_num_feats: int
    # mask generator parameters
    msk_kernel_size: int
    msk_num_feats: int
    msk_num_hidden_feats: int
    msk_num_layers: int
    msk_num_stacks: int
    msk_activate: str


class ConvTasNet(ConvTasNetTorch):
    def __init__(self, config: ConvTasNetConfig):
        super().__init__(**config)


class ConvTasNetFactory(ModelFactory):
    @staticmethod
    def _get_config():
        return ConvTasNetConfig

    @staticmethod
    def _get_object(config: ModelConfig):
        return ConvTasNet(check_config_entries(config, ConvTasNetConfig))
