# This file is an extraction of needed modules from asteroid with added type annotations and additional error handling
#       Additionally fixes minor bugs
# Original repo: https://github.com/asteroid-team/asteroid
import inspect
import math
from typing import Any, cast, Optional, Type

import torch
import torch.nn as nn


def pad_x_to_y(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Right-pad or right-trim first argument to have same size as second argument

    Args:
        x (torch.Tensor): Tensor to be padded.
        y (torch.Tensor): Tensor to pad `x` to.
        axis (int): Axis to pad on.

    Returns:
        torch.Tensor, `x` padded to match `y`'s shape.
    """
    inp_len = y.shape[-1]
    output_len = x.shape[-1]
    return nn.functional.pad(x, [0, inp_len - output_len])


def has_arg(fn: Any, name: str):
    """Checks if a callable accepts a given keyword argument.

    Args:
        fn (callable): Callable to inspect.
        name (str): Check if ``fn`` can be called with ``name`` as a keyword
            argument.

    Returns:
        bool: whether ``fn`` accepts a ``name`` keyword argument.
    """
    signature = inspect.signature(fn)
    parameter = signature.parameters.get(name)
    if parameter is None:
        return False
    return parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def z_norm(x: torch.Tensor, dims: list[int], eps: float = 1e-8):
    mean = x.mean(dim=dims, keepdim=True)
    var2 = torch.var(x, dim=dims, keepdim=True, unbiased=False)
    value = (x - mean) / torch.sqrt((var2 + eps))
    return value


def _glob_norm(x: torch.Tensor, eps: float = 1e-8):
    dims = list(range(1, x.dim()))
    return z_norm(x, dims, eps)


def _feat_glob_norm(x: torch.Tensor, eps: float = 1e-8):
    dims = list(range(2, x.dim()))
    return z_norm(x, dims, eps)


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size: int):
        super().__init__()

        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x: torch.Tensor):
        """Assumes input of size `[batch, chanel, *]`."""
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x: torch.Tensor, EPS: float = 1e-8):
        """Applies forward pass.

        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        value = _glob_norm(x, eps=EPS)
        return self.apply_gain_and_bias(value)


class ChanLN(_LayerNorm):
    """Channel-wise Layer Normalization (chanLN)."""

    def forward(self, x: torch.Tensor, EPS: float = 1e-8):
        """Applies forward pass.

        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: chanLN_x `[batch, chan, *]`
        """
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        return self.apply_gain_and_bias((x - mean) / (var + EPS).sqrt())


class CumLN(_LayerNorm):
    """Cumulative Global layer normalization(cumLN)."""

    def forward(self, x: torch.Tensor, EPS: float = 1e-8):
        """

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, channels, length]`
        Returns:
             :class:`torch.Tensor`: cumLN_x `[batch, channels, length]`
        """
        _, chan, spec_len = x.size()
        cum_sum = torch.cumsum(x.sum(1, keepdim=True), dim=-1)
        cum_pow_sum = torch.cumsum(x.pow(2).sum(1, keepdim=True), dim=-1)
        cnt = torch.arange(
            start=chan,
            end=chan * (spec_len + 1),
            step=chan,
            dtype=x.dtype,
            device=x.device,
        ).view(1, 1, -1)
        cum_mean = cum_sum / cnt
        cum_var = cum_pow_sum / cnt - cum_mean.pow(2)
        return self.apply_gain_and_bias((x - cum_mean) / (cum_var + EPS).sqrt())


class FeatsGlobLN(_LayerNorm):
    """Feature-wise global Layer Normalization (FeatsGlobLN).
    Applies normalization over frames for each channel."""

    def forward(self, x: torch.Tensor, EPS: float = 1e-8):
        """Applies forward pass.

        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): `[batch, chan, time]`

        Returns:
            :class:`torch.Tensor`: chanLN_x `[batch, chan, time]`
        """
        value = _feat_glob_norm(x, eps=EPS)
        return self.apply_gain_and_bias(value)


class BatchNorm(nn.modules.batchnorm._BatchNorm):
    """Wrapper class for pytorch BatchNorm1D and BatchNorm2D"""

    def _check_input_dim(self, input: torch.Tensor):
        if input.dim() < 2 or input.dim() > 4:
            raise ValueError(
                "expected 4D or 3D input (got {}D input)".format(input.dim())
            )


class DualPathProcessing(nn.Module):
    """
    Perform Dual-Path processing via overlap-add as in DPRNN [1].

    Args:
        chunk_size (int): Size of segmenting window.
        hop_size (int): segmentation hop size.

    References
        [1] Yi Luo, Zhuo Chen and Takuya Yoshioka. "Dual-path RNN: efficient
        long sequence modeling for time-domain single-channel speech separation"
        https://arxiv.org/abs/1910.06379
    """

    def __init__(self, chunk_size: int, hop_size: int):
        super().__init__()

        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.n_orig_frames = None

    def unfold(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Unfold the feature tensor from $(batch, channels, time)$ to
        $(batch, channels, chunksize, nchunks)$.

        Args:
            x (:class:`torch.Tensor`): feature tensor of shape $(batch, channels, time)$.

        Returns:
            :class:`torch.Tensor`: spliced feature tensor of shape
            $(batch, channels, chunksize, nchunks)$.

        """
        # x is (batch, chan, frames)
        batch, chan, _ = x.size()
        assert x.ndim == 3
        self.n_orig_frames = x.shape[-1]
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )

        return unfolded.reshape(
            batch, chan, self.chunk_size, -1
        )  # (batch, chan, chunk_size, n_chunks)

    def fold(self, x: torch.Tensor, output_size: Optional[int] = None) -> torch.Tensor:
        r"""
        Folds back the spliced feature tensor.
        Input shape $(batch, channels, chunksize, nchunks)$ to original shape
        $(batch, channels, time)$ using overlap-add.

        Args:
            x (:class:`torch.Tensor`): spliced feature tensor of shape
                $(batch, channels, chunksize, nchunks)$.
            output_size (int, optional): sequence length of original feature tensor.
                If None, the original length cached by the previous call of
                :meth:`unfold` will be used.

        Returns:
            :class:`torch.Tensor`:  feature tensor of shape $(batch, channels, time)$.

        .. note:: `fold` caches the original length of the input.

        """
        if self.n_orig_frames is None:
            raise RuntimeError("Invalid state reached")
        output_size = output_size if output_size is not None else self.n_orig_frames
        # x is (batch, chan, chunk_size, n_chunks)
        batch, chan, _, n_chunks = x.size()
        to_unfold = x.reshape(batch, chan * self.chunk_size, n_chunks)
        x = torch.nn.functional.fold(
            to_unfold,
            (output_size, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )

        # force float div for torch jit
        x /= float(self.chunk_size) / self.hop_size

        return x.reshape(batch, chan, self.n_orig_frames)

    @staticmethod
    def intra_process(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
        r"""Performs intra-chunk processing.

        Args:
            x (:class:`torch.Tensor`): spliced feature tensor of shape
                (batch, channels, chunk_size, n_chunks).
            module (:class:`torch.nn.Module`): module one wish to apply to each chunk
                of the spliced feature tensor.

        Returns:
            :class:`torch.Tensor`: processed spliced feature tensor of shape
            $(batch, channels, chunksize, nchunks)$.

        .. note:: the module should have the channel first convention and accept
            a 3D tensor of shape $(batch, channels, time)$.
        """

        # x is (batch, channels, chunk_size, n_chunks)
        batch, channels, chunk_size, n_chunks = x.size()
        # we reshape to batch*chunk_size, channels, n_chunks
        x = (
            x.transpose(1, -1)
            .reshape(batch * n_chunks, chunk_size, channels)
            .transpose(1, -1)
        )
        x = module(x)
        x = (
            x.reshape(batch, n_chunks, channels, chunk_size)
            .transpose(1, -1)
            .transpose(1, 2)
        )
        return x

    @staticmethod
    def inter_process(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
        r"""Performs inter-chunk processing.

        Args:
            x (:class:`torch.Tensor`): spliced feature tensor of shape
                $(batch, channels, chunksize, nchunks)$.
            module (:class:`torch.nn.Module`): module one wish to apply between
                each chunk of the spliced feature tensor.


        Returns:
            x (:class:`torch.Tensor`): processed spliced feature tensor of shape
            $(batch, channels, chunksize, nchunks)$.

        .. note:: the module should have the channel first convention and accept
            a 3D tensor of shape $(batch, channels, time)$.
        """

        batch, channels, chunk_size, n_chunks = x.size()
        x = x.transpose(1, 2).reshape(batch * chunk_size, channels, n_chunks)
        x = module(x)
        x = x.reshape(batch, chunk_size, channels, n_chunks).transpose(1, 2)
        return x


class ImprovedTransformedLayer(nn.Module):
    """
    Improved Transformer module as used in [1].
    It is Multi-Head self-attention followed by LSTM, activation and linear projection layer.

    Args:
        embed_dim (int): Number of input channels.
        n_heads (int): Number of attention heads.
        dim_ff (int): Number of neurons in the RNNs cell state.
            Defaults to 256. RNN here replaces standard FF linear layer in plain Transformer.
        dropout (float, optional): Dropout ratio, must be in [0,1].
        activation (str, optional): activation function applied at the output of RNN.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        norm (str, optional): Type of normalization to use.

    References
        [1] Chen, Jingjing, Qirong Mao, and Dong Liu. "Dual-Path Transformer
        Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation."
        arXiv (2020).
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dim_ff: int,
        dropout: float = 0.0,
        activation: Type[nn.Module] = nn.ReLU,
        bidirectional: bool = True,
        norm: Type[nn.Module] = GlobLN,
    ):
        super().__init__()

        self.mha = nn.modules.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.recurrent = nn.LSTM(
            embed_dim, dim_ff, bidirectional=bidirectional, batch_first=True
        )
        ff_inner_dim = 2 * dim_ff if bidirectional else dim_ff
        self.linear = nn.Linear(ff_inner_dim, embed_dim)
        self.activation = activation()
        self.norm_mha = norm(embed_dim)
        self.norm_ff = norm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tomha = x.permute(2, 0, 1)
        # x is batch, channels, seq_len
        # mha is seq_len, batch, channels
        # self-attention is applied
        out = self.mha(tomha, tomha, tomha)[0]
        x = self.dropout(out.permute(1, 2, 0)) + x
        x = self.norm_mha(x)

        # lstm is applied
        out = self.linear(
            self.dropout(self.activation(self.recurrent(x.transpose(1, -1))[0]))
        )
        x = self.dropout(out.transpose(1, -1)) + x
        return self.norm_ff(x)


class DPTransformer(nn.Module):
    """Dual-path Transformer introduced in [1].

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        n_heads (int): Number of attention heads.
        ff_hid (int): Number of neurons in the RNNs cell state.
            Defaults to 256.
        chunk_size (int): window size of overlap and add processing.
            Defaults to 100.
        hop_size (int or None): hop size (stride) of overlap and add processing.
            Default to `chunk_size // 2` (50% overlap).
        n_repeats (int): Number of repeats. Defaults to 6.
        norm_type (str, optional): Type of normalization to use.
        ff_activation (str, optional): activation function applied at the output of RNN.
        mask_act (str, optional): Which non-linear function to generate mask.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        dropout (float, optional): Dropout ratio, must be in [0,1].

    References
        [1] Chen, Jingjing, Qirong Mao, and Dong Liu. "Dual-Path Transformer
        Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation."
        arXiv (2020).
    """

    def __init__(
        self,
        in_chan: int,
        n_src: int,
        n_heads: int = 4,
        ff_hid: int = 256,
        chunk_size: int = 100,
        hop_size: Optional[int] = None,
        n_repeats: int = 6,
        norm_type: Type[nn.Module] = GlobLN,
        ff_activation: Type[nn.Module] = nn.ReLU,
        mask_act: Type[nn.Module] = nn.ReLU,
        bidirectional: bool = True,
        dropout: float = 0,
    ):
        super().__init__()

        self.in_chan = in_chan
        self.n_src = n_src
        self.n_heads = n_heads
        self.ff_hid = ff_hid
        self.chunk_size = chunk_size
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        self.n_src = n_src
        self.norm_type = norm_type
        self.ff_activation = ff_activation
        self.mask_act = mask_act
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.mha_in_dim = math.ceil(self.in_chan / self.n_heads) * self.n_heads
        if self.in_chan % self.n_heads != 0:
            raise RuntimeError(
                f"DPTransformer input dim ({self.in_chan}) is not a multiple of the number of "
                f"heads ({self.n_heads}). Adding extra linear layer at input to accomodate "
                f"(size [{self.in_chan} x {self.mha_in_dim}])"
            )

        self.in_norm = norm_type(self.mha_in_dim)
        self.ola = DualPathProcessing(self.chunk_size, self.hop_size)

        # Succession of DPRNNBlocks.
        self.layers = nn.ModuleList([])
        for _ in range(self.n_repeats):
            self.layers.append(
                nn.ModuleList(
                    [
                        ImprovedTransformedLayer(
                            self.mha_in_dim,
                            self.n_heads,
                            self.ff_hid,
                            self.dropout,
                            self.ff_activation,
                            True,
                            self.norm_type,
                        ),
                        ImprovedTransformedLayer(
                            self.mha_in_dim,
                            self.n_heads,
                            self.ff_hid,
                            self.dropout,
                            self.ff_activation,
                            self.bidirectional,
                            self.norm_type,
                        ),
                    ]
                )
            )
        net_out_conv = nn.Conv2d(self.mha_in_dim, n_src * self.in_chan, 1)
        self.first_out = nn.Sequential(nn.PReLU(), net_out_conv)
        # Gating and masking in 2D space (after fold)
        self.net_out = nn.Sequential(
            nn.Conv1d(self.in_chan, self.in_chan, 1), nn.Tanh()
        )
        self.net_gate = nn.Sequential(
            nn.Conv1d(self.in_chan, self.in_chan, 1), nn.Sigmoid()
        )

        # Get activation function.
        mask_nl_class = mask_act
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, "dim"):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w: torch.Tensor) -> torch.Tensor:
        r"""Forward.

        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$

        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, nsrc, nfilters, nframes)$
        """
        mixture_w = self.in_norm(mixture_w)  # [batch, bn_chan, n_frames]
        n_orig_frames = mixture_w.shape[-1]

        mixture_w = self.ola.unfold(mixture_w)
        batch, _, self.chunk_size, n_chunks = mixture_w.size()

        for layer_idx in range(len(self.layers)):
            module_list = self.layers[layer_idx]
            intra, inter = cast(nn.ModuleList, module_list)
            mixture_w = self.ola.intra_process(mixture_w, intra)
            mixture_w = self.ola.inter_process(mixture_w, inter)

        output: torch.Tensor = self.first_out(mixture_w)
        output = output.reshape(
            batch * self.n_src, self.in_chan, self.chunk_size, n_chunks
        )
        output = self.ola.fold(output, output_size=n_orig_frames)

        output = self.net_out(output) * self.net_gate(output)
        # Compute mask
        output = output.reshape(batch, self.n_src, self.in_chan, -1)
        est_mask = self.output_act(output)
        return est_mask


class BaseEncoderMaskerDecoder(nn.Module):
    """Base class for encoder-masker-decoder separation models.

    Args:
        encoder (Encoder): Encoder instance.
        masker (nn.Module): masker network.
        decoder (Decoder): Decoder instance.
        encoder_activation (Optional[str], optional): Activation to apply after encoder.
            See ``asteroid.masknn.activations`` for valid values.
    """

    def __init__(
        self,
        encoder: nn.Module,
        masker: nn.Module,
        decoder: nn.Module,
        encoder_activation: Type[nn.Module],
    ):
        super().__init__()

        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder
        self.encoder_activation = encoder_activation
        self.enc_activation = encoder_activation()

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Enc/Mask/Dec model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
        # Real forward
        tf_rep = self.forward_encoder(wav)
        est_masks = self.forward_masker(tf_rep)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)

        reconstructed = pad_x_to_y(decoded, wav)
        return reconstructed

    def forward_encoder(self, wav: torch.Tensor) -> torch.Tensor:
        """Computes time-frequency representation of `wav`.

        Args:
            wav (torch.Tensor): waveform tensor in 3D shape, time last.

        Returns:
            torch.Tensor, of shape (batch, feat, seq).
        """
        tf_rep = self.encoder(wav)
        return self.enc_activation(tf_rep)

    def forward_masker(self, tf_rep: torch.Tensor) -> torch.Tensor:
        """Estimates masks from time-frequency representation.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation in (batch,
                feat, seq).

        Returns:
            torch.Tensor: Estimated masks
        """
        return self.masker(tf_rep)

    def apply_masks(
        self, tf_rep: torch.Tensor, est_masks: torch.Tensor
    ) -> torch.Tensor:
        """Applies masks to time-frequency representation.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation in (batch,
                feat, seq) shape.
            est_masks (torch.Tensor): Estimated masks.

        Returns:
            torch.Tensor: Masked time-frequency representations.
        """
        return est_masks * tf_rep.unsqueeze(1)

    def forward_decoder(self, masked_tf_rep: torch.Tensor) -> torch.Tensor:
        """Reconstructs time-domain waveforms from masked representations.

        Args:
            masked_tf_rep (torch.Tensor): Masked time-frequency representation.

        Returns:
            torch.Tensor: Time-domain waveforms.
        """
        return self.decoder(masked_tf_rep)


class SingleRNN(nn.Module):
    """Module for a RNN block.

    Inspired from https://github.com/yluo42/TAC/blob/master/utility/models.py
    Licensed under CC BY-NC-SA 3.0 US.

    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        input_size (int): Dimension of the input feature. The input should have
            shape [batch, seq_len, input_size].
        hidden_size (int): Dimension of the hidden state.
        n_layers (int, optional): Number of layers used in RNN. Default is 1.
        dropout (float, optional): Dropout ratio. Default is 0.
        bidirectional (bool, optional): Whether the RNN layers are
            bidirectional. Default is ``False``.
    """

    def __init__(
        self,
        rnn_type: Type[nn.RNN | nn.LSTM | nn.GRU],
        input_size: int,
        hidden_size: int,
        n_layers: int = 1,
        dropout: float = 0,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn = rnn_type(
            input_size,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bool(bidirectional),
        )

    @property
    def output_size(self):
        return self.hidden_size * (2 if self.bidirectional else 1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Input shape [batch, seq, feats]"""
        # self.rnn.flatten_parameters()  # Enables faster multi-GPU training.
        output = inp
        rnn_output, _ = self.rnn(output)
        return rnn_output


class LSTMMasker(nn.Module):
    """LSTM mask network introduced in [1], without skip connections.

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        out_chan  (int or None): Number of bins in the estimated masks.
            Defaults to `in_chan`.
        rnn_type (str, optional): Type of RNN used. Choose between ``'RNN'``,
            ``'LSTM'`` and ``'GRU'``.
        n_layers (int, optional): Number of layers in each RNN.
        hid_size (int): Number of neurons in the RNNs cell state.
        mask_act (str, optional): Which non-linear function to generate mask.
        bidirectional (bool, optional): Whether to use BiLSTM
        dropout (float, optional): Dropout ratio, must be in [0,1].

    References
        [1]: Yi Luo et al. "Real-time Single-channel Dereverberation and Separation
        with Time-domain Audio Separation Network", Interspeech 2018
    """

    def __init__(
        self,
        in_chan: int,
        n_src: int,
        out_chan: Optional[int] = None,
        rnn_type: Type[nn.RNN | nn.LSTM | nn.GRU] = nn.LSTM,
        n_layers: int = 4,
        hid_size: int = 512,
        dropout: float = 0.3,
        mask_act: Type[nn.Module] = nn.Sigmoid,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan is not None else in_chan
        self.out_chan = out_chan
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        self.hid_size = hid_size
        self.dropout = dropout
        self.mask_act = mask_act
        self.bidirectional = bidirectional

        # For softmax, feed the source dimension.
        if has_arg(mask_act, "dim"):
            self.output_act = mask_act(dim=1)
        else:
            self.output_act = mask_act()

        # Create TasNet masker
        out_size = hid_size * (int(bidirectional) + 1)
        if bidirectional:
            self.bn_layer = GlobLN(in_chan)
        else:
            self.bn_layer = CumLN(in_chan)
        self.masker = nn.Sequential(
            SingleRNN(
                self.rnn_type,
                in_chan,
                hidden_size=hid_size,
                n_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout,
            ),
            nn.Linear(out_size, self.n_src * out_chan),
            self.output_act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        to_sep: torch.Tensor = self.bn_layer(x)
        est_masks: torch.Tensor = self.masker(to_sep.transpose(-1, -2))
        est_masks = est_masks.transpose(-1, -2)
        est_masks = est_masks.view(batch_size, self.n_src, self.out_chan, -1)
        return est_masks


class _Chop1d(nn.Module):
    """To ensure the output length is the same as the input."""

    def __init__(self, chop_size: int):
        super().__init__()

        self.chop_size = chop_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., : -self.chop_size].contiguous()


class BaseConv1DBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        skip_out_chan: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        norm_type: Type[nn.Module] = GlobLN,
        causal: bool = False,
    ):
        super().__init__()

        self.skip_out_chan = skip_out_chan
        conv_norm = norm_type
        in_conv1d = nn.Conv1d(in_chan, hid_chan, 1)
        depth_conv1d = nn.Conv1d(
            hid_chan,
            hid_chan,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hid_chan,
        )
        if causal:
            depth_conv1d = nn.Sequential(depth_conv1d, _Chop1d(padding))
        self.shared_block = nn.Sequential(
            in_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan),
            depth_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan),
        )
        self.skip_conv = nn.Conv1d(hid_chan, skip_out_chan, 1)


class Conv1DBlock(BaseConv1DBlock):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        skip_out_chan: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        norm_type: Type[nn.Module] = GlobLN,
        causal: bool = False,
    ):
        super().__init__(
            in_chan,
            hid_chan,
            skip_out_chan,
            kernel_size,
            padding,
            dilation,
            norm_type,
            causal,
        )
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        r"""Input shape $(batch, feats, seq)$."""
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        skip_out = self.skip_conv(shared_out)
        return res_out, skip_out


class FinalConv1DBlock(BaseConv1DBlock):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        skip_out_chan: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        norm_type: Type[nn.Module] = GlobLN,
        causal: bool = False,
    ):
        super().__init__(
            in_chan,
            hid_chan,
            skip_out_chan,
            kernel_size,
            padding,
            dilation,
            norm_type,
            causal,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        r"""Input shape $(batch, feats, seq)$."""
        shared_out = self.shared_block(x)
        skip_out = self.skip_conv(shared_out)
        return torch.zeros(1), skip_out


class TDConvNet(nn.Module):
    """Temporal Convolutional network used in ConvTasnet.

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
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
        causal (bool, optional) : Whether or not the convolutions are causal.

    References
        [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
        for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
        https://arxiv.org/abs/1809.07454
    """

    def __init__(
        self,
        in_chan: int,
        n_src: int,
        out_chan: Optional[int] = None,
        n_blocks: int = 8,
        n_repeats: int = 3,
        bn_chan: int = 128,
        hid_chan: int = 512,
        skip_chan: int = 128,
        conv_kernel_size: int = 3,
        norm_type: Type[nn.Module] = GlobLN,
        mask_act: Type[nn.Module] = nn.ReLU,
        causal: bool = False,
    ):
        super().__init__()

        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        self.conv_kernel_size = conv_kernel_size
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.causal = causal

        layer_norm = norm_type(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for i in range(n_repeats):
            for x in range(n_blocks):
                if not causal:
                    padding = (conv_kernel_size - 1) * 2**x // 2
                else:
                    padding = (conv_kernel_size - 1) * 2**x
                if i == n_repeats - 1 and x == n_blocks - 1:
                    self.TCN.append(
                        FinalConv1DBlock(
                            bn_chan,
                            hid_chan,
                            skip_chan,
                            conv_kernel_size,
                            padding=padding,
                            dilation=2**x,
                            norm_type=norm_type,
                            causal=causal,
                        )
                    )
                else:
                    self.TCN.append(
                        Conv1DBlock(
                            bn_chan,
                            hid_chan,
                            skip_chan,
                            conv_kernel_size,
                            padding=padding,
                            dilation=2**x,
                            norm_type=norm_type,
                            causal=causal,
                        )
                    )
        mask_conv_inp = skip_chan
        mask_conv = nn.Conv1d(mask_conv_inp, n_src * out_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        # Get activation function.
        mask_nl_class = mask_act
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, "dim"):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w: torch.Tensor) -> torch.Tensor:
        r"""Forward.

        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$

        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, nsrc, nfilters, nframes)$
        """
        batch, _, n_frames = mixture_w.size()
        output: torch.Tensor = self.bottleneck(mixture_w)
        skip_connection = torch.tensor([0.0], device=output.device)
        for layer in self.TCN:
            # Common to w. skip and w.o skip architectures
            tcn_out = layer(output)
            residual, skip = tcn_out
            skip_connection = skip_connection + skip
            output = output + residual
        # Use residual output when no skip connection
        mask_inp = skip_connection
        score: torch.Tensor = self.mask_net(mask_inp)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        est_mask = self.output_act(score)
        return est_mask
