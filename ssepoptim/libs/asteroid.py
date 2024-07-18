# This file is an extraction of needed modules from asteroid with added type annotations and additional error handling
# Original repo: https://github.com/asteroid-team/asteroid
import inspect
import math
import warnings
from typing import Any, Optional, Type

import torch
import torch.nn as nn


def _shape_reconstructed(
    reconstructed: torch.Tensor, size: torch.Tensor
) -> torch.Tensor:
    """Reshape `reconstructed` to have same size as `size`

    Args:
        reconstructed (torch.Tensor): Reconstructed waveform
        size (torch.Tensor): Size of desired waveform

    Returns:
        torch.Tensor: Reshaped waveform

    """
    if len(size) == 1:
        return reconstructed.squeeze(0)
    return reconstructed


def pad_x_to_y(x: torch.Tensor, y: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Right-pad or right-trim first argument to have same size as second argument

    Args:
        x (torch.Tensor): Tensor to be padded.
        y (torch.Tensor): Tensor to pad `x` to.
        axis (int): Axis to pad on.

    Returns:
        torch.Tensor, `x` padded to match `y`'s shape.
    """
    if axis != -1:
        raise NotImplementedError
    inp_len = y.shape[axis]
    output_len = x.shape[axis]
    return nn.functional.pad(x, [0, inp_len - output_len])


def jitable_shape(tensor: torch.Tensor) -> torch.Tensor:
    """Gets shape of ``tensor`` as ``torch.Tensor`` type for jit compiler

    .. note::
        Returning ``tensor.shape`` of ``tensor.size()`` directly is not torchscript
        compatible as return type would not be supported.

    Args:
        tensor (torch.Tensor): Tensor

    Returns:
        torch.Tensor: Shape of ``tensor``
    """
    return torch.tensor(tensor.shape)


def _unsqueeze_to_3d(x: torch.Tensor) -> torch.Tensor:
    """Normalize shape of `x` to [batch, n_chan, time]."""
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    elif x.ndim == 2:
        return x.unsqueeze(1)
    else:
        return x


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
    dims: list[int] = torch.arange(1, len(x.shape)).tolist()
    return z_norm(x, dims, eps)


def _feat_glob_norm(x: torch.Tensor, eps: float = 1e-8):
    dims: list[int] = torch.arange(2, len(x.shape)).tolist()
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
        norm: Type[nn.Module] = GlobLN,
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
        self.norm = norm
        self.ff_activation = ff_activation
        self.mask_act = mask_act
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.mha_in_dim = math.ceil(self.in_chan / self.n_heads) * self.n_heads
        if self.in_chan % self.n_heads != 0:
            warnings.warn(
                f"DPTransformer input dim ({self.in_chan}) is not a multiple of the number of "
                f"heads ({self.n_heads}). Adding extra linear layer at input to accomodate "
                f"(size [{self.in_chan} x {self.mha_in_dim}])"
            )
            self.input_layer = nn.Linear(self.in_chan, self.mha_in_dim)
        else:
            self.input_layer = None

        self.in_norm = norm(self.mha_in_dim)
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
                            self.norm,
                        ),
                        ImprovedTransformedLayer(
                            self.mha_in_dim,
                            self.n_heads,
                            self.ff_hid,
                            self.dropout,
                            self.ff_activation,
                            self.bidirectional,
                            self.norm,
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
        if self.input_layer is not None:
            mixture_w = self.input_layer(mixture_w.transpose(1, 2)).transpose(1, 2)
        mixture_w = self.in_norm(mixture_w)  # [batch, bn_chan, n_frames]
        n_orig_frames = mixture_w.shape[-1]

        mixture_w = self.ola.unfold(mixture_w)
        batch, _, self.chunk_size, n_chunks = mixture_w.size()

        for layer_idx in range(len(self.layers)):
            intra, inter = self.layers[layer_idx]
            mixture_w = self.ola.intra_process(mixture_w, intra)
            mixture_w = self.ola.inter_process(mixture_w, inter)

        output = self.first_out(mixture_w)
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
        encoder_activation: Optional[Type[nn.Module]] = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder
        self.encoder_activation = encoder_activation
        self.enc_activation = (encoder_activation or nn.Identity)()

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Enc/Mask/Dec model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)

        # Real forward
        tf_rep = self.forward_encoder(wav)
        est_masks = self.forward_masker(tf_rep)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)

        reconstructed = pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape)

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
