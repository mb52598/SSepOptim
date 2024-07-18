# This file is an extraction of needed modules from speechbrain with added type annotations and additional error handling
# Original repo: https://github.com/speechbrain/speechbrain
import copy
import math
from typing import Any, Optional, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

shape_type = list[int] | torch.Size


class LayerNorm(nn.Module):
    """Applies layer normalization to the input tensor.

    Arguments
    ---------
    input_size : int
        The expected size of the dimension to be normalized.
    input_shape : tuple
        The expected shape of the input.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    elementwise_affine : bool
        If True, this module has learnable per-element affine parameters
        initialized to ones (for weights) and zeros (for biases).

    Example
    -------
    >>> input = torch.randn(100, 101, 128)
    >>> norm = LayerNorm(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 101, 128])
    """

    def __init__(
        self,
        input_size: int | shape_type,
        input_shape: Optional[shape_type] = None,
        eps: float = 1e-05,
        elementwise_affine: bool = True,
    ):
        super().__init__()

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if input_shape is not None:
            input_size = input_shape[2:]

        self.norm = torch.nn.LayerNorm(
            input_size,
            eps=self.eps,
            elementwise_affine=self.elementwise_affine,
        )

    def forward(self, x: torch.Tensor):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channels)
            input to normalize. 3d or 4d tensors are expected.

        Returns
        -------
        The normalized outputs.
        """
        return self.norm(x)


def get_padding_elem(L_in: int, stride: int, kernel_size: int, dilation: int):
    """This function computes the number of elements to add for zero-padding.

    Arguments
    ---------
    L_in : int
    stride: int
    kernel_size : int
    dilation : int

    Returns
    -------
    padding : int
        The size of the padding to be added
    """
    if stride > 1:
        padding = [math.floor(kernel_size / 2), math.floor(kernel_size / 2)]

    else:
        L_out = math.floor((L_in - dilation * (kernel_size - 1) - 1) / stride) + 1
        padding = [
            math.floor((L_in - L_out) / 2),
            math.floor((L_in - L_out) / 2),
        ]
    return padding


class Conv1d(nn.Module):
    """This function implements 1d convolution.

    Arguments
    ---------
    out_channels : int
        It is the number of output channels.
    kernel_size : int
        Kernel size of the convolutional filters.
    input_shape : tuple
        The shape of the input. Alternatively use ``in_channels``.
    in_channels : int
        The number of input channels. Alternatively use ``input_shape``.
    stride : int
        Stride factor of the convolutional filters. When the stride factor > 1,
        a decimation in time is performed.
    dilation : int
        Dilation factor of the convolutional filters.
    padding : str
        (same, valid, causal). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is the same as the input shape.
        "causal" results in causal (dilated) convolutions.
    groups : int
        Number of blocked connections from input channels to output channels.
    bias : bool
        Whether to add a bias term to convolution operation.
    padding_mode : str
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    skip_transpose : bool
        If False, uses batch x time x channel convention of speechbrain.
        If True, uses batch x channel x time convention.
    weight_norm : bool
        If True, use weight normalization,
        to be removed with self.remove_weight_norm() at inference
    conv_init : str
        Weight initialization for the convolution network
    default_padding: str or int
        This sets the default padding mode that will be used by the pytorch Conv1d backend.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 40, 16])
    >>> cnn_1d = Conv1d(
    ...     input_shape=inp_tensor.shape, out_channels=8, kernel_size=5
    ... )
    >>> out_tensor = cnn_1d(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 40, 8])
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        input_shape: Optional[shape_type] = None,
        in_channels: Optional[int] = None,
        stride: int = 1,
        dilation: int = 1,
        padding: str = "same",
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "reflect",
        skip_transpose: bool = False,
        weight_norm: bool = False,
        conv_init: Optional[str] = None,
        default_padding: str | int = 0,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.unsqueeze = False
        self.skip_transpose = skip_transpose

        if input_shape is None and in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")

        if in_channels is None:
            if input_shape is None:
                raise RuntimeError("Unexpected error happened")

            in_channels = self._check_input_shape(input_shape)

        self.in_channels = in_channels

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=default_padding,
            groups=groups,
            bias=bias,
        )

        if conv_init == "kaiming":
            nn.init.kaiming_normal_(self.conv.weight)
        elif conv_init == "zero":
            nn.init.zeros_(self.conv.weight)
        elif conv_init == "normal":
            nn.init.normal_(self.conv.weight, std=1e-6)

        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x: torch.Tensor):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 2d or 4d tensors are expected.

        Returns
        -------
        wx : torch.Tensor
            The convolved outputs.
        """
        if not self.skip_transpose:
            x = x.transpose(1, -1)

        if self.unsqueeze:
            x = x.unsqueeze(1)

        if self.padding == "same":
            x = self._manage_padding(x, self.kernel_size, self.dilation, self.stride)

        elif self.padding == "causal":
            num_pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (num_pad, 0))

        elif self.padding == "valid":
            pass

        else:
            raise ValueError(
                "Padding must be 'same', 'valid' or 'causal'. Got " + self.padding
            )

        wx = self.conv(x)

        if self.unsqueeze:
            wx = wx.squeeze(1)

        if not self.skip_transpose:
            wx = wx.transpose(1, -1)

        return wx

    def _manage_padding(
        self, x: torch.Tensor, kernel_size: int, dilation: int, stride: int
    ):
        """This function performs zero-padding on the time axis
        such that their lengths is unchanged after the convolution.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        kernel_size : int
            Size of kernel.
        dilation : int
            Dilation used.
        stride : int
            Stride.

        Returns
        -------
        x : torch.Tensor
            The padded outputs.
        """

        # Detecting input shape
        L_in = self.in_channels

        # Time padding
        padding = get_padding_elem(L_in, stride, kernel_size, dilation)

        # Applying padding
        x = F.pad(x, padding, mode=self.padding_mode)

        return x

    def _check_input_shape(self, shape: shape_type):
        """Checks the input shape and returns the number of input channels."""

        if len(shape) == 2:
            self.unsqueeze = True
            in_channels = 1
        elif self.skip_transpose:
            in_channels = shape[1]
        elif len(shape) == 3:
            in_channels = shape[2]
        else:
            raise ValueError("conv1d expects 2d, 3d inputs. Got " + str(len(shape)))

        # Kernel size must be odd
        if not self.padding == "valid" and self.kernel_size % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )

        return in_channels

    def remove_weight_norm(self):
        """Removes weight normalization at inference if used during training."""
        self.conv = nn.utils.remove_weight_norm(self.conv)


class MultiheadAttention(nn.Module):
    """The class is a wrapper of MultiHead Attention for torch.nn.MultiHeadAttention.

    Reference: https://pytorch.org/docs/stable/nn.html

    Arguments
    ---------
    nhead : int
        parallel attention heads.
    d_model : int
        The size of the model layers.
    dropout : float
        a Dropout layer on attn_output_weights (default: 0.0).
    bias : bool
        add bias as module parameter (default: True).
    add_bias_kv : bool
        add bias to the key and value sequences at dim=0.
    add_zero_attn : bool
        add a new batch of zeros to the key and value sequences at dim=1.
    kdim : int
        total number of features in key (default: None).
    vdim : int
        total number of features in value (default: None).

    Example
    -------
    >>> inputs = torch.rand([8, 60, 512])
    >>> net = MultiheadAttention(nhead=8, d_model=inputs.shape[-1])
    >>> outputs, attn = net(inputs, inputs, inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        nhead: int,
        d_model: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
    ):
        super().__init__()

        self.att = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = True,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """Compute attention.

        Arguments
        ---------
        query : torch.Tensor
            (B, L, E) where L is the target sequence length,
            B is the batch size, E is the embedding dimension.
        key : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        value : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        attn_mask : torch.Tensor, optional
            2D mask (L, S) where L is the target sequence length, S is
            the source sequence length.
            3D mask (N*num_heads, L, S) where N is the batch
            size, L is the target sequence length, S is the source sequence
            length. attn_mask ensure that position i is allowed to attend the
            unmasked positions. If a ByteTensor is provided, the non-zero
            positions are not allowed to attend while the zero positions will
            be unchanged. If a BoolTensor is provided, positions with True is
            not allowed to attend while False values will be unchanged. If a
            FloatTensor is provided, it will be added to the attention weight.
        key_padding_mask : torch.Tensor, optional
            (B, S) where B is the batch size, S is the source sequence
            length. If a ByteTensor is provided, the non-zero positions will
            be ignored while the position with the zero positions will be
            unchanged. If a BoolTensor is provided, the positions with the
            value of True will be ignored while the position with the value
            of False will be unchanged.
        return_attn_weights : bool, optional
            True to additionally return the attention weights, False otherwise.
        pos_embs : torch.Tensor, optional
            Positional embeddings added to the attention map of shape (L, S, E) or (L, S, 1).

        Returns
        -------
        attn_output : torch.Tensor
            (B, L, E) where L is the target sequence length, B is the
            batch size, E is the embedding dimension.
        attn_output_weights : torch.Tensor
            (B, L, S) where B is the batch size, L is the target
            sequence length, S is the source sequence length.
            This is returned only if `return_attn_weights=True` (True by default).
        """
        # give tensors of shape (time, batch, fea)
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        # this will be legit because of https://github.com/pytorch/pytorch/blob/5288d05cfdda85c46c4df84617fa7f37c21b10b3/torch/nn/functional.py#L4946
        # we can inject relative learnable pos embeddings directly in MHA via the attn_mask
        if pos_embs is not None:
            if attn_mask is not None:
                attn_mask += pos_embs
            else:
                attn_mask = pos_embs

        output, attention_weights = self.att(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn_weights,
        )

        # reshape the output back to (batch, time, fea)
        output = output.permute(1, 0, 2)

        if return_attn_weights:
            return output, attention_weights

        return output


class RelPosMHAXL(nn.Module):
    """This class implements the relative multihead implementation similar to that in Transformer XL
    https://arxiv.org/pdf/1901.02860.pdf

    Arguments
    ---------
    embed_dim : int
        Size of the encoder feature vectors from which keys and values are computed.
    num_heads: int
        Number of attention heads.
    dropout : float, optional
        Dropout rate.
    vbias: bool, optional
        Whether to use bias for computing value.
    vdim: int, optional
        Size for value. Default is embed_dim (Note each head is embed_dim // num_heads).
    mask_pos_future: bool, optional
        Whether to mask future positional encodings values.
        Must be true for causal applications e.g. decoder.

    Example
    -------
    >>> inputs = torch.rand([6, 60, 512])
    >>> pos_emb = torch.rand([1, 2*60-1, 512])
    >>> net = RelPosMHAXL(num_heads=8, embed_dim=inputs.shape[-1])
    >>> outputs, attn = net(inputs, inputs, inputs, pos_emb)
    >>> outputs.shape
    torch.Size([6, 60, 512])
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        vbias: Optional[bool] = False,
        vdim: Optional[int] = None,
        mask_pos_future: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.vdim == embed_dim
        self.mask_pos_future = mask_pos_future
        self.vbias = vbias

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.vhead_dim = self.vdim // num_heads

        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        assert (
            self.vhead_dim * num_heads == self.vdim
        ), "vdim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.qk_proj_weight = nn.Parameter(torch.empty(2 * embed_dim, embed_dim))
            self.v_proj_weight = nn.Parameter(torch.empty(self.vdim, embed_dim))
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))

        if vbias:
            self.value_bias_weight = nn.Parameter(torch.empty(self.vdim))
        else:
            self.vbias = None

        self.dropout_att = nn.Dropout(dropout)
        self.out_proj = nn.Linear(self.vdim, embed_dim)

        self.linear_pos = nn.Linear(embed_dim, embed_dim, bias=False)

        self.pos_bias_u = nn.Parameter(torch.empty(self.head_dim, self.num_heads))
        self.pos_bias_v = nn.Parameter(torch.empty(self.head_dim, self.num_heads))

        if next(self.parameters()).dtype == torch.float16:
            self.attn_fill_value = -65000
        else:
            self.attn_fill_value = -float("inf")

        self._reset_parameters()
        self.scale = 1 / math.sqrt(self.embed_dim)

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            torch.nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            torch.nn.init.xavier_uniform_(self.qk_proj_weight)
            torch.nn.init.xavier_uniform_(self.v_proj_weight)

        if self.vbias is not None:
            torch.nn.init.constant_(self.value_bias_weight, 0.0)

        # positional biases
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x: torch.Tensor):
        """Relative shift implementation."""
        # batch, head, time1, 2*time1-1.

        b, h, qlen, pos_len = x.size()  # (b, h, t1, t2)
        # need to add a column of zeros on the left side of last dimension to perform the relative shifting
        x = torch.nn.functional.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, h, -1, qlen)  # (b, h, t2+1, t1)
        # need to drop the first row
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # (b, h, t1, t2)

        # cspell:ignore tril
        if self.mask_pos_future:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x[..., : pos_len // 2 + 1]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_embs: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = True,
    ):
        """Compute attention.

        Arguments
        ---------
        query : torch.Tensor
            (B, L, E) where L is the target sequence length,
            B is the batch size, E is the embedding dimension.
        key : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        value : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        pos_embs : torch.Tensor
            bidirectional sinusoidal positional embedding tensor (1, 2*S-1, E) where S is the max length between source and target sequence lengths,
            and E is the embedding dimension.
        key_padding_mask : torch.Tensor
            (B, S) where B is the batch size, S is the source sequence
            length. If a ByteTensor is provided, the non-zero positions will
            be ignored while the position with the zero positions will be
            unchanged. If a BoolTensor is provided, the positions with the
            value of True will be ignored while the position with the value
            of False will be unchanged.
        attn_mask : torch.Tensor
            2D mask (L, S) where L is the target sequence length, S is
            the source sequence length.
            3D mask (N*num_heads, L, S) where N is the batch
            size, L is the target sequence length, S is the source sequence
            length. attn_mask ensure that position i is allowed to attend the
            unmasked positions. If a ByteTensor is provided, the non-zero
            positions are not allowed to attend while the zero positions will
            be unchanged. If a BoolTensor is provided, positions with True is
            not allowed to attend while False values will be unchanged. If a
            FloatTensor is provided, it will be added to the attention weight.
        return_attn_weights : bool
            Whether to additionally return the attention weights.

        Returns
        -------
        out : torch.Tensor
            (B, L, E) where L is the target sequence length, B is the
            batch size, E is the embedding dimension.
        attn_score : torch.Tensor
            (B, L, S) where B is the batch size, L is the target
            sequence length, S is the source sequence length.
        """

        # query, key and value are of shape batch, time, embed_dim
        bsz = query.shape[0]
        klen = key.shape[1]
        qlen = query.shape[1]

        if self._qkv_same_embed_dim:
            # self-attention
            if (query is key or torch.equal(query, key)) and (
                key is value or torch.equal(key, value)
            ):
                query, key, value = (
                    nn.functional.linear(query, self.in_proj_weight)
                    .view(bsz, -1, self.num_heads, self.head_dim * 3)
                    .chunk(3, dim=-1)
                )
            else:
                qweight, kweight, vweight = self.in_proj_weight.chunk(3, dim=0)
                query = nn.functional.linear(query, qweight).view(
                    bsz, -1, self.num_heads, self.head_dim
                )
                key = nn.functional.linear(key, kweight).view(
                    bsz, -1, self.num_heads, self.head_dim
                )
                value = nn.functional.linear(value, vweight).view(
                    bsz, -1, self.num_heads, self.head_dim
                )
        else:
            raise NotImplementedError
            query, key = (
                nn.functional.linear(query, self.qk_proj_weight)
                .view(bsz, -1, self.num_heads, self.head_dim * 2)
                .chunk(2, dim=-1)
            )
            value = nn.functional.linear(value, self.v_proj_weight).view(
                bsz, -1, self.num_heads, self.vhead_dim
            )

        if self.vbias is not None:
            value = value + self.value_bias_weight.view(
                1, 1, self.num_heads, self.vhead_dim
            )

        p_k = self.linear_pos(pos_embs).view(1, -1, self.num_heads, self.head_dim)
        # (batch, head, klen, d_k)

        q_with_bias_u = (
            query + self.pos_bias_u.view(1, 1, self.num_heads, self.head_dim)
        ).transpose(1, 2)
        # (batch, head, qlen, d_k)
        q_with_bias_v = (
            query + self.pos_bias_v.view(1, 1, self.num_heads, self.head_dim)
        ).transpose(1, 2)

        # Moved the `* self.scale` mul from after the `attn_score` sum to prior
        # to the matmul in order to lower overflow risks on fp16.
        # This change is inspired by the following paper, but no other changes
        # were ported from there so far.
        # ref: E.T.: Re-Thinking Self-Attention for Transformer Models on GPUs
        # https://asherliu.github.io/docs/sc21a.pdf

        # (batch, head, qlen, klen)
        matrix_ac = torch.matmul(q_with_bias_u * self.scale, key.permute(0, 2, 3, 1))
        # (batch, num_heads, klen, 2*klen-1)
        matrix_bd = torch.matmul(q_with_bias_v * self.scale, p_k.permute(0, 2, 3, 1))
        matrix_bd = self.rel_shift(matrix_bd)  # shifting trick

        # if klen != qlen:
        #   import ipdb
        #  ipdb.set_trace(

        attn_score = matrix_ac + matrix_bd  # already scaled above

        # compute attention probability
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.view(1, 1, qlen, klen)
            else:
                attn_mask = attn_mask.view(-1, self.num_heads, qlen, klen)

            if attn_mask.dtype == torch.bool:
                attn_score = attn_score.masked_fill(attn_mask, self.attn_fill_value)
            else:
                attn_score += attn_mask

        if key_padding_mask is not None:
            attn_score = attn_score.masked_fill(
                key_padding_mask.view(bsz, 1, 1, klen),
                self.attn_fill_value,
            )

        attn_score = F.softmax(attn_score, dim=-1, dtype=torch.float32)
        attn_score = self.dropout_att(attn_score)

        # it is possible for us to hit full NaN when using chunked training
        # so reapply masks, except with 0.0 instead as we are after the softmax
        # because -inf would output 0.0 regardless anyway
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_score = attn_score.masked_fill(attn_mask, 0.0)
            else:
                # NOTE: the above fix is not implemented for this case as
                # summing the mask with NaN would still result in NaN
                pass

        if key_padding_mask is not None:
            attn_score = attn_score.masked_fill(
                key_padding_mask.view(bsz, 1, 1, klen),
                0.0,
            )

        x = torch.matmul(attn_score, value.transpose(1, 2))  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(bsz, -1, self.vhead_dim * self.num_heads)
        )  # (batch, time1, d_model)

        out = self.out_proj(x)
        if return_attn_weights:
            return out, attn_score
        return out


class PositionalwiseFeedForward(nn.Module):
    """The class implements the positional-wise feed forward module in
    “Attention Is All You Need”.

    Arguments
    ---------
    d_ffn: int
        Hidden layer size.
    input_shape : tuple, optional
        Expected shape of the input. Alternatively use ``input_size``.
    input_size : int, optional
        Expected size of the input. Alternatively use ``input_shape``.
    dropout: float, optional
        Dropout rate.
    activation: torch.nn.Module, optional
        activation functions to be applied (Recommendation: ReLU, GELU).

    Example
    -------
    >>> inputs = torch.rand([8, 60, 512])
    >>> net = PositionalwiseFeedForward(256, input_size=inputs.shape[-1])
    >>> outputs = net(inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn: int,
        input_shape: Optional[shape_type] = None,
        input_size: Optional[int] = None,
        dropout: float = 0.0,
        activation: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None and input_shape is not None:
            input_size = input_shape[-1]

        if input_size is None:
            raise RuntimeError("Unexpected error happened")

        self.ffn = nn.Sequential(
            nn.Linear(input_size, d_ffn),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, input_size),
        )

    def forward(self, x: torch.Tensor):
        """Applies PositionalwiseFeedForward to the input tensor x."""
        # give a tensor of shape (time, batch, fea)
        x = x.permute(1, 0, 2)
        x = self.ffn(x)

        # reshape the output back to (batch, time, fea)
        x = x.permute(1, 0, 2)

        return x


class ParallelMLPs(nn.Module):
    """Class that implements the MultiHead HyperMixer or HyperConformer.

    Arguments
    ----------
    input_size : int
        Dimension of the linear layers
    hidden_size: int
        Dimension of the hidden layer
    output_size : int
        Dimension of the HyperNetwork
    num_mlps : int
        Number of heads, akin to heads in MultiHeadAttention
    keep_output_size : bool, optional
        Set whether to keep the same output size independent of number of heads
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_mlps: int = 1,
        keep_output_size: bool = True,
    ) -> None:
        super().__init__()

        if output_size is None:
            output_size = input_size

        self.original_in_size = input_size
        self.original_out_size = output_size

        assert input_size % num_mlps == 0
        assert output_size % num_mlps == 0
        assert hidden_size % num_mlps == 0
        input_size = input_size // num_mlps

        if not keep_output_size:
            output_size = output_size // num_mlps
        hidden_size = hidden_size // num_mlps

        self.input_size = input_size
        self.output_size = output_size

        self.num_mlps = num_mlps

        # set the weights and biases parameters
        self.fc1_weights = nn.Parameter(torch.empty(num_mlps, hidden_size, input_size))
        self.fc1_biases = nn.Parameter(torch.empty(num_mlps, hidden_size))
        self.fc2_weights = nn.Parameter(torch.empty(num_mlps, output_size, hidden_size))
        self.fc2_biases = nn.Parameter(torch.empty(num_mlps, output_size))

        # initialize the weights and biases
        nn.init.xavier_uniform_(self.fc1_weights, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.fc1_biases, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.fc2_weights, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.fc2_biases, gain=math.sqrt(2.0))

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor):
        """Performs the forward computation of multi parallel MLPs.

        Arguments
        ----------
        x : tensor
            Input tensor

        Outputs
        -------
        x : torch.Tensor
            return output tensor
        """

        # x [bsize, seq_len, num_features]
        bsize = x.size(0)
        seq_len = x.size(1)

        # Reshape the input tensor to match the number of parallel MLPs and their input size
        x = x.reshape((bsize, seq_len, self.num_mlps, self.input_size))

        # Perform the first linear transformation and add bias
        # Using einsum so we can do it for multiple MLPs in parallel
        x = torch.einsum(
            "blmf,mhf->bmlh", x, self.fc1_weights
        ) + self.fc1_biases.unsqueeze(0).unsqueeze(2)

        # Apply activation function and perform the second linear transformation and add bias
        x = self.activation(x)
        x = torch.einsum(
            "bmlh,mfh->bmlf", x, self.fc2_weights
        ) + self.fc2_biases.unsqueeze(0).unsqueeze(2)

        return x


class HyperNetwork(nn.Module):
    """This class implements The HyperNetwork. It is an approach of using a one network,
    also known as a hypernetwork, to generate the weights for another network.
    Here, it is used to generate the labels of linear layers.

    Reference: https://arxiv.org/abs/1609.09106

    Arguments
    ----------
    input_output_dim : int
        Dimension of the linear layers
    hypernet_size:
        Dimension of the HyperNetwork
    tied : bool, optional
        Define whether weights of layer 1 and layer 2 are shared
    num_heads: int, optional
        Number of heads, akin to heads in MultiHeadAttention
    keep_output_size: bool, optional
        Set whether to keep the same output size independent of number of heads
    """

    def __init__(
        self,
        input_output_dim: int,
        hypernet_size: int,
        tied: bool = False,
        num_heads: int = 1,
        keep_output_size: bool = True,
    ) -> None:
        super().__init__()

        # Define whether the two linear layers have tied weights
        self.tied = tied
        self.w1_gen = ParallelMLPs(
            input_output_dim,
            input_output_dim,
            output_size=hypernet_size,
            num_mlps=num_heads,
            keep_output_size=keep_output_size,
        )
        if self.tied:
            self.w2_gen = self.w1_gen
        else:
            self.w2_gen = ParallelMLPs(
                input_output_dim,
                input_output_dim,
                output_size=hypernet_size,
                num_mlps=num_heads,
                keep_output_size=keep_output_size,
            )

    def forward(self, input_tensor: torch.Tensor):
        """Forward computation for a HyperNetwork.

        Arguments
        ----------
        input_tensor : [batchsize, max_positions, d]
            The HyperNetwork is supposed to generate an MLP of the form W_2(GELU(W1 x)), where
            W1 : N -> k and W2 : k -> N, so it has to return tensors W1 and W2

        Outputs
        -------
        W1 : torch.Tensor
            Generated weights of Layer 1
        W2 : torch.Tensor
            Generated weights of Layer 2
        """
        w1 = self.w1_gen(input_tensor)
        if self.tied:
            w2 = w1
        else:
            w2 = self.w2_gen(input_tensor)

        return w1, w2


class HyperMixing(nn.Module):
    """This class implements multi-head HyperMixing.
    It is an implementation of the token-mixing component in HyperMixer, a linear
    time drop-in replacement for self-attention. In contrast to the original HyperMixer,
    this module supports multiple heads, which improves the expressiveness of the model
    while decreasing the number of parameters.

    Reference: https://arxiv.org/abs/2203.03691

    Arguments
    ---------
    input_output_dim : int
        number of features in keys, queries, and values
    hypernet_size : int
        determines the size of the hidden layer of the token-mixing MLP.
    tied : bool
        If True, then the generated weight matrices of the token-mixing MLP are tied.
    num_heads : int
        parallel token-mixing MLPs.
    fix_tm_hidden_size : bool
        If True, the hidden-layer size is equal to hypernet_size rather than hypernet_size / num_heads.
    max_length : int
        Maximum number of input tokens. Needed for generating sufficiently large position embeddings.

    Example
    -------
    >>> import torch
    >>> inputs = torch.rand([8, 60, 512])
    >>> net = HyperMixing(512, 2048, num_heads=8)
    >>> outputs, attn = net(inputs, inputs, inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        input_output_dim: int,
        hypernet_size: int,
        tied: bool = False,
        num_heads: int = 1,
        fix_tm_hidden_size: bool = False,
        max_length: int = 3000,
    ) -> None:
        super().__init__()

        self.input_output_dim = input_output_dim
        self.hyper = HyperNetwork(
            input_output_dim,
            hypernet_size,
            tied=tied,
            num_heads=num_heads,
            keep_output_size=fix_tm_hidden_size,
        )
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(input_output_dim)
        self.num_heads = num_heads

        # add pos encoding
        self.positional_encoding = PositionalEncoding(input_output_dim, max_length)

    def _mlp_pass_from_components(
        self,
        out: torch.Tensor,
        W1: torch.Tensor,
        W2: torch.Tensor,
        activation: nn.Module,
    ):
        """function to stick MLP1 together manually"""
        out = torch.bmm(out, W1)
        out = activation(out)
        out = torch.bmm(out, W2.transpose(1, 2))
        return out

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights: Optional[bool] = True,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        The signature of this method is deliberately chosen to be the same as for
        sb.nnet.attention.MultiHeadAttention for compatibility within SpeechBrain.

        NOTE: key, value, attn_mask and pos_embs have no effect. Query is used for
        all three. Thus, the module should only be used to replace self-attention at the moment.

        Arguments
        ----------
        query : torch.Tensor
            (B, L, E) where L is the target sequence length,
            B is the batch size, E is the embedding dimension.
        key : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
            Currently unused. All
        value : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
            Currently unused.
        attn_mask : torch.Tensor, optional
            NOTE: Currently has NO effect.
        key_padding_mask : torch.Tensor, optional
            (B, S) where B is the batch size, S is the source sequence
            length. If a ByteTensor is provided, the non-zero positions will
            be ignored while the position with the zero positions will be
            unchanged. If a BoolTensor is provided, the positions with the
            value of True will be ignored while the position with the value
            of False will be unchanged.
        return_attn_weights: torch.Tensor, optional
            NOTE: Currently has NO effect.
        pos_embs: torch.Tensor, optional
            NOTE: Currently has NO effect.

        Outputs
        -------
        attn_output : torch.Tensor
            (B, L, E) where L is the target sequence length, B is the
            batch size, E is the embedding dimension.
        attn_output_weights : torch.Tensor
            (B, L, S) where B is the batch size, L is the target
            sequence length, S is the source sequence length.
            NOTE: always returns all zeros.
        """

        float_mask = None

        # NOTE: We are ignoring keys and values, because HyperMixing can only be used in the encoder atm (where it's all the same)
        out = query

        bsize = out.size(0)
        seq_len = out.size(1)

        if key_padding_mask is not None:
            float_mask = torch.logical_not(key_padding_mask).unsqueeze(-1).float()
            out = out * float_mask

        # add position embedding before passing to hypernetwork
        hyp_input = out + self.positional_encoding(out)
        w1, w2 = self.hyper(
            hyp_input
        )  # [bsize, num_heads, seq_len, hypernet_size // num_heads]

        if key_padding_mask is not None:
            if float_mask is None:
                raise RuntimeError("Unexpected error happened")
            # mask the weights
            w1 = w1 * float_mask.unsqueeze(1)
            w2 = w2 * float_mask.unsqueeze(1)

        # reshape the num_heads into the batch dimension for parallelizing
        out = out.transpose(1, 2)  # [bsize, input_output_dim, seq_len]
        out = out.reshape(
            (
                bsize * self.num_heads,
                self.input_output_dim // self.num_heads,
                seq_len,
            )
        )  # [bsize * num_heads, input_output_dim // num_heads, seq_len]
        w1 = w1.reshape((bsize * self.num_heads, seq_len, -1))
        w2 = w2.reshape((bsize * self.num_heads, seq_len, -1))

        # we stick the token-mixing MLP together manually
        out = self._mlp_pass_from_components(out, w1, w2, self.activation)

        # concatenate heads
        out = out.reshape((bsize, self.input_output_dim, seq_len))

        # transpose back
        out = out.transpose(1, 2)

        # apply layer norm on outputs of the TM-MLP
        out = self.layer_norm(out)

        dummy_att_weights = torch.zeros((bsize, seq_len, seq_len), device=out.device)
        return out, dummy_att_weights


class TransformerEncoderLayer(nn.Module):
    """This is an implementation of self-attention encoder layer.

    Arguments
    ---------
    d_ffn: int, optional
        The dimension of the feedforward network model hidden layer.
    nhead: int
        The number of heads in the multi-head attention models (default=8).
    d_model: int
        The number of expected features in the encoder/decoder inputs (default=512).
    kdim: int, optional
        Dimension of the key.
    vdim: int, optional
        Dimension of the value.
    dropout: int, optional
        The dropout value.
    activation: torch.nn.Module, optional
        The activation function for Feed-Forward Network layer,
        e.g., relu or gelu or swish.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.
    ffn_type: str
        type of ffn: regularFFN/1dcnn
    ffn_cnn_kernel_size_list: list of int
        kernel size of 2 1d-convs if ffn_type is 1dcnn
    causal: bool, optional
        Whether the encoder should be causal or not (the decoder is always causal).
        If causal the Conformer convolutional layer is causal.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoderLayer(512, 8, d_model=512)
    >>> output = net(x)
    >>> output[0].shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn: int,
        nhead: int,
        d_model: int,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        dropout: float = 0.0,
        activation: Type[nn.Module] = nn.ReLU,
        normalize_before: bool = False,
        attention_type: str = "regularMHA",
        ffn_type: str = "regularFFN",
        ffn_cnn_kernel_size_list: list[int] = [3, 3],
        causal: bool = False,
    ):
        super().__init__()

        if attention_type == "regularMHA":
            self.self_att = MultiheadAttention(
                nhead=nhead,
                d_model=d_model,
                dropout=dropout,
                kdim=kdim,
                vdim=vdim,
            )

        elif attention_type == "RelPosMHAXL":
            self.self_att = RelPosMHAXL(d_model, nhead, dropout, mask_pos_future=causal)
        elif attention_type == "hypermixing":
            self.self_att = HyperMixing(
                input_output_dim=d_model,
                hypernet_size=d_ffn,
                tied=False,
                num_heads=nhead,
                fix_tm_hidden_size=False,
            )

        if ffn_type == "regularFFN":
            self.pos_ffn = PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            )
        elif ffn_type == "1dcnn":
            self.pos_ffn = nn.Sequential(
                Conv1d(
                    in_channels=d_model,
                    out_channels=d_ffn,
                    kernel_size=ffn_cnn_kernel_size_list[0],
                    padding="causal" if causal else "same",
                ),
                nn.ReLU(),
                Conv1d(
                    in_channels=d_ffn,
                    out_channels=d_model,
                    kernel_size=ffn_cnn_kernel_size_list[1],
                    padding="causal" if causal else "same",
                ),
            )

        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.normalize_before = normalize_before
        self.pos_ffn_type = ffn_type

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ---------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_mask : torch.Tensor
            The mask for the src query for each example in the batch.
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src keys for each example in the batch.
        pos_embs: torch.Tensor, optional
            The positional embeddings tensor.

        Returns
        -------
        output : torch.Tensor
            The output of the transformer encoder layer.
        """

        if self.normalize_before:
            src1 = self.norm1(src)
        else:
            src1 = src

        output, self_attn = self.self_att(
            src1,
            src1,
            src1,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs,
        )

        # add & norm
        src = src + self.dropout1(output)
        if not self.normalize_before:
            src = self.norm1(src)

        if self.normalize_before:
            src1 = self.norm2(src)
        else:
            src1 = src
        output = self.pos_ffn(src1)

        # add & norm
        output = src + self.dropout2(output)
        if not self.normalize_before:
            output = self.norm2(output)
        return output, self_attn


class TransformerEncoder(nn.Module):
    """This class implements the transformer encoder.

    Arguments
    ---------
    num_layers : int
        Number of transformer layers to include.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    input_shape : tuple
        Expected shape of the input.
    d_model : int
        The dimension of the input embedding.
    kdim : int
        Dimension for key (Optional).
    vdim : int
        Dimension for value (Optional).
    dropout : float
        Dropout for the encoder (Optional).
    activation: torch.nn.Module, optional
        The activation function for Feed-Forward Network layer,
        e.g., relu or gelu or swish.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    causal: bool, optional
        Whether the encoder should be causal or not (the decoder is always causal).
        If causal the Conformer convolutional layer is causal.
    layerdrop_prob: float
        The probability to drop an entire layer
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.
    ffn_type: str
        type of ffn: regularFFN/1dcnn
    ffn_cnn_kernel_size_list: list of int
        conv kernel size of 2 1d-convs if ffn_type is 1dcnn

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoder(1, 8, 512, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers: int,
        nhead: int,
        d_ffn: int,
        input_shape: Optional[shape_type] = None,
        d_model: Optional[int] = None,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        dropout: float = 0.0,
        activation: Type[nn.Module] = nn.ReLU,
        normalize_before: bool = False,
        causal: bool = False,
        layerdrop_prob: float = 0.0,
        attention_type: str = "regularMHA",
        ffn_type: str = "regularFFN",
        ffn_cnn_kernel_size_list: list[int] = [3, 3],
    ):
        super().__init__()

        if d_model is None:
            raise RuntimeError("Unexpected error happened")

        self.layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=causal,
                    attention_type=attention_type,
                    ffn_type=ffn_type,
                    ffn_cnn_kernel_size_list=ffn_cnn_kernel_size_list,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model, eps=1e-6)
        self.layerdrop_prob = layerdrop_prob
        self.rng = np.random.default_rng()

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
        dynchunktrain_config: None = None,
    ):
        """
        Arguments
        ---------
        src : torch.Tensor
            The sequence to the encoder layer (required).
        src_mask : torch.Tensor
            The mask for the src sequence (optional).
        src_key_padding_mask : torch.Tensor
            The mask for the src keys per batch (optional).
        pos_embs : torch.Tensor
            The positional embedding tensor
        dynchunktrain_config : config
            Not supported for this encoder.

        Returns
        -------
        output : torch.Tensor
            The output of the transformer.
        attention_lst : list
            The attention values.
        """
        assert (
            dynchunktrain_config is None
        ), "Dynamic Chunk Training unsupported for this encoder"

        output = src
        if self.layerdrop_prob > 0.0:
            keep_probs = self.rng.random(len(self.layers))
        else:
            keep_probs = None
        attention_lst: list[Any] = []
        enc_layer: nn.Module
        for i, enc_layer in enumerate(self.layers):
            if (
                not self.training
                or self.layerdrop_prob == 0.0
                or (keep_probs is not None and keep_probs[i] > self.layerdrop_prob)
            ):
                output, attention = enc_layer(
                    output,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    pos_embs=pos_embs,
                )

                attention_lst.append(attention)
        output = self.norm(output)
        return output, attention_lst


class SBTransformerBlock(nn.Module):
    """A wrapper for the SpeechBrain implementation of the transformer encoder.

    Arguments
    ---------
    num_layers : int
        Number of layers.
    d_model : int
        Dimensionality of the representation.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Dimensionality of positional feed forward.
    input_shape : tuple
        Shape of input.
    kdim : int
        Dimension of the key (Optional).
    vdim : int
        Dimension of the value (Optional).
    dropout : float
        Dropout rate.
    activation : str
        Activation function.
    use_positional_encoding : bool
        If true we use a positional encoding.
    norm_before: bool
        Use normalization before transformations.

    Example
    ---------
    >>> x = torch.randn(10, 100, 64)
    >>> block = SBTransformerBlock(1, 64, 8)
    >>> x = block(x)
    >>> x.shape
    torch.Size([10, 100, 64])
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        d_ffn: int = 2048,
        input_shape: Optional[shape_type] = None,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_positional_encoding: bool = False,
        norm_before: bool = False,
        attention_type: str = "regularMHA",
    ):
        super().__init__()

        self.use_positional_encoding = use_positional_encoding

        if activation == "relu":
            activation_class = nn.ReLU
        elif activation == "gelu":
            activation_class = nn.GELU
        else:
            raise ValueError("unknown activation")

        self.mdl = TransformerEncoder(
            num_layers=num_layers,
            nhead=nhead,
            d_ffn=d_ffn,
            input_shape=input_shape,
            d_model=d_model,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            activation=activation_class,
            normalize_before=norm_before,
            attention_type=attention_type,
        )

        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(input_size=d_model)

    def forward(self, x: torch.Tensor):
        """Returns the transformed output.

        Arguments
        ---------
        x : torch.Tensor
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   L = time points
                   N = number of filters

        """
        if self.use_positional_encoding:
            pos_enc = self.pos_enc(x)
            return self.mdl(x + pos_enc)[0]
        else:
            return self.mdl(x)[0]


class SBRNNBlock(nn.Module):
    """RNNBlock for the dual path pipeline.

    Arguments
    ---------
    input_size : int
        Dimensionality of the input features.
    hidden_channels : int
        Dimensionality of the latent layer of the rnn.
    num_layers : int
        Number of the rnn layers.
    rnn_type : str
        Type of the the rnn cell.
    dropout : float
        Dropout rate
    bidirectional : bool
        If True, bidirectional.

    Example
    ---------
    >>> x = torch.randn(10, 100, 64)
    >>> rnn = SBRNNBlock(64, 100, 1, bidirectional=True)
    >>> x = rnn(x)
    >>> x.shape
    torch.Size([10, 100, 200])
    """

    def __init__(
        self,
        input_size: int,
        hidden_channels: int,
        num_layers: int,
        rnn_type: str = "LSTM",
        dropout: int = 0,
        bidirectional: bool = True,
    ):
        super().__init__()

        raise NotImplementedError("SBRNNBlock is not implemented")

    def forward(self, x: torch.Tensor):
        """Returns the transformed output.

        Arguments
        ---------
        x : torch.Tensor
            [B, L, N]
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """

        raise NotImplementedError("SBRNNBlock.forward is not implemented")


class GlobalLayerNorm(nn.Module):
    """Calculate Global Layer Normalization.

    Arguments
    ---------
       dim : (int or list or torch.Size)
           Input shape from an expected input of size.
       eps : float
           A value added to the denominator for numerical stability.
       elementwise_affine : bool
          A boolean value that when set to True,
          this module has learnable per-element affine parameters
          initialized to ones (for weights) and zeros (for biases).

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> GLN = GlobalLayerNorm(10, 3)
    >>> x_norm = GLN(x)
    """

    def __init__(
        self,
        dim: int,
        shape: int,
        eps: float = 1e-8,
        elementwise_affine: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of size [N, C, K, S] or [N, C, L].
        """
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)

        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    """Calculate Cumulative Layer Normalization.

       Arguments
       ---------
       dim : int
        Dimension that you want to normalize.
       elementwise_affine : True
        Learnable per-element affine parameters.

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> CLN = CumulativeLayerNorm(10)
    >>> x_norm = CLN(x)
    """

    def __init__(self, dim: int, elementwise_affine: bool = True, eps: float = 1e-8):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, input: torch.Tensor):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor size [N, C, K, S] or [N, C, L]
        """
        # input: N x C x K x S or N x C x L
        # N x K x S x C
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            # N x K x S x C == only channel norm
            input = super().forward(input)
            # N x C x K x S
            input = input.permute(0, 3, 1, 2).contiguous()
        if input.dim() == 3:
            input = torch.transpose(input, 1, 2)
            # N x L x C == only channel norm
            input = super().forward(input)
            # N x C x L
            input = torch.transpose(input, 1, 2)
        return input


def select_norm(norm: Optional[str], dim: int, shape: int, eps: float = 1e-8):
    """Just a wrapper to select the normalization type."""

    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True, eps=eps)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True, eps=eps)
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=eps)
    else:
        return nn.BatchNorm1d(dim)


class Encoder(nn.Module):
    """Convolutional Encoder Layer.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.

    Example
    -------
    >>> x = torch.randn(2, 1000)
    >>> encoder = Encoder(kernel_size=4, out_channels=64)
    >>> h = encoder(x)
    >>> h.shape
    torch.Size([2, 64, 499])
    """

    def __init__(
        self, kernel_size: int = 2, out_channels: int = 64, in_channels: int = 1
    ):
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor):
        """Return the encoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, L].
        Return
        ------
        x : torch.Tensor
            Encoded tensor with dimensionality [B, N, T_out].

        where B = Batchsize
              L = Number of timepoints
              N = Number of filters
              T_out = Number of timepoints at the output of the encoder
        """
        # B x L -> B x 1 x L
        if self.in_channels == 1:
            x = torch.unsqueeze(x, dim=1)
        # B x 1 x L -> B x N x T_out
        x = self.conv1d(x)
        x = F.relu(x)

        return x


class Decoder(nn.ConvTranspose1d):
    """A decoder layer that consists of ConvTranspose1d.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.


    Example
    ---------
    >>> x = torch.randn(2, 100, 1000)
    >>> decoder = Decoder(kernel_size=4, in_channels=100, out_channels=1)
    >>> h = decoder(x)
    >>> h.shape
    torch.Size([2, 1003])
    """

    def forward(self, input: torch.Tensor, output_size: Optional[list[int]] = None):
        """Return the decoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, N, L].
                where, B = Batchsize,
                       N = number of filters
                       L = time points
        """

        if input.dim() not in [2, 3]:
            raise RuntimeError("{} accept 3/4D tensor as input".format(self.__name__))
        input = super().forward(
            input if input.dim() == 3 else torch.unsqueeze(input, 1), output_size
        )

        if torch.squeeze(input).dim() == 1:
            input = torch.squeeze(input, dim=1)
        else:
            input = torch.squeeze(input)
        return input


class Linear(torch.nn.Module):
    """Computes a linear transformation y = wx + b.

    Arguments
    ---------
    n_neurons : int
        It is the number of output neurons (i.e, the dimensionality of the
        output).
    input_shape : tuple
        It is the shape of the input tensor.
    input_size : int
        Size of the input tensor.
    bias : bool
        If True, the additive bias b is adopted.
    max_norm : float
        weight max-norm.
    combine_dims : bool
        If True and the input is 4D, combine 3rd and 4th dimensions of input.

    Example
    -------
    >>> inputs = torch.rand(10, 50, 40)
    >>> lin_t = Linear(input_shape=(10, 50, 40), n_neurons=100)
    >>> output = lin_t(inputs)
    >>> output.shape
    torch.Size([10, 50, 100])
    """

    def __init__(
        self,
        n_neurons: int,
        input_shape: Optional[shape_type] = None,
        input_size: Optional[int] = None,
        bias: bool = True,
        max_norm: Optional[float] = None,
        combine_dims: bool = False,
    ):
        super().__init__()

        self.max_norm = max_norm
        self.combine_dims = combine_dims

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None and input_shape is not None:
            input_size = input_shape[-1]
            if len(input_shape) == 4 and self.combine_dims:
                input_size = input_shape[2] * input_shape[3]

        if input_size is None:
            raise RuntimeError("Unexpected error happened")

        # Weights are initialized following pytorch approach
        self.w = nn.Linear(input_size, n_neurons, bias=bias)

    def forward(self, x: torch.Tensor):
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input to transform linearly.

        Returns
        -------
        wx : torch.Tensor
            The linearly transformed outputs.
        """
        if x.ndim == 4 and self.combine_dims:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        if self.max_norm is not None:
            self.w.weight.data = torch.renorm(
                self.w.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )

        wx = self.w(x)

        return wx


class Dual_Computation_Block(nn.Module):
    """Computation block for dual-path processing.

    Arguments
    ---------
    intra_mdl : torch.nn.module
        Model to process within the chunks.
     inter_mdl : torch.nn.module
        Model to process across the chunks.
     out_channels : int
        Dimensionality of inter/intra model.
     norm : str
        Normalization type.
     skip_around_intra : bool
        Skip connection around the intra layer.
     linear_layer_after_inter_intra : bool
        Linear layer or not after inter or intra.

    Example
    ---------
        >>> intra_block = SBTransformerBlock(1, 64, 8)
        >>> inter_block = SBTransformerBlock(1, 64, 8)
        >>> dual_comp_block = Dual_Computation_Block(intra_block, inter_block, 64)
        >>> x = torch.randn(10, 64, 100, 10)
        >>> x = dual_comp_block(x)
        >>> x.shape
        torch.Size([10, 64, 100, 10])
    """

    def __init__(
        self,
        intra_mdl: nn.Module,
        inter_mdl: nn.Module,
        out_channels: int,
        norm: Optional[str] = "ln",
        skip_around_intra: bool = True,
        linear_layer_after_inter_intra: bool = True,
    ):
        super().__init__()

        self.intra_mdl = intra_mdl
        self.inter_mdl = inter_mdl
        self.skip_around_intra = skip_around_intra
        self.linear_layer_after_inter_intra = linear_layer_after_inter_intra

        # Norm
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 4)
            self.inter_norm = select_norm(norm, out_channels, 4)

        # Linear
        if linear_layer_after_inter_intra:
            if isinstance(intra_mdl, SBRNNBlock):
                self.intra_linear = Linear(
                    out_channels, input_size=2 * intra_mdl.mdl.rnn.hidden_size
                )
            else:
                self.intra_linear = Linear(out_channels, input_size=out_channels)

            if isinstance(inter_mdl, SBRNNBlock):
                self.inter_linear = Linear(
                    out_channels, input_size=2 * intra_mdl.mdl.rnn.hidden_size
                )
            else:
                self.inter_linear = Linear(out_channels, input_size=out_channels)

    def forward(self, x: torch.Tensor):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, K, S].


        Return
        ---------
        out: torch.Tensor
            Output tensor of dimension [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
        """
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        # [BS, K, H]

        intra = self.intra_mdl(intra)

        # [BS, K, N]
        if self.linear_layer_after_inter_intra:
            intra = self.intra_linear(intra)

        # [B, S, K, N]
        intra = intra.view(B, S, K, N)
        # [B, N, K, S]
        intra = intra.permute(0, 3, 2, 1).contiguous()
        if self.norm is not None:
            intra = self.intra_norm(intra)

        # [B, N, K, S]
        if self.skip_around_intra:
            intra = intra + x

        # inter RNN
        # [BK, S, N]
        inter = intra.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        # [BK, S, H]
        inter = self.inter_mdl(inter)

        # [BK, S, N]
        if self.linear_layer_after_inter_intra:
            inter = self.inter_linear(inter)

        # [B, K, S, N]
        inter = inter.view(B, K, S, N)
        # [B, N, K, S]
        inter = inter.permute(0, 3, 1, 2).contiguous()
        if self.norm is not None:
            inter = self.inter_norm(inter)
        # [B, N, K, S]
        out = inter + intra

        return out


class PositionalEncoding(nn.Module):
    """This class implements the absolute sinusoidal positional encoding function.
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))

    Arguments
    ---------
    input_size: int
        Embedding dimension.
    max_len : int, optional
        Max length of the input sequences (default 2500).

    Example
    -------
    >>> a = torch.rand((8, 120, 512))
    >>> enc = PositionalEncoding(input_size=a.shape[-1])
    >>> b = enc(a)
    >>> b.shape
    torch.Size([1, 120, 512])
    """

    def __init__(self, input_size: int, max_len: int = 2500):
        super().__init__()

        if input_size % 2 != 0:
            raise ValueError(
                f"Cannot use sin/cos positional encoding with odd channels (got channels={input_size})"
            )
        self.max_len = max_len
        pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, input_size, 2).float() * -(math.log(10000.0) / input_size)
        )

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """
        Arguments
        ---------
        x : torch.Tensor
            Input feature shape (batch, time, fea)

        Returns
        -------
        The positional encoding.
        """
        return self.pe[:, : x.size(1)].clone().detach()


class Dual_Path_Model(nn.Module):
    """The dual path model which is the basis for dualpathrnn, sepformer, dptnet.

    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the intra and inter blocks.
    intra_model : torch.nn.module
        Model to process within the chunks.
    inter_model : torch.nn.module
        model to process across the chunks,
    num_layers : int
        Number of layers of Dual Computation Block.
    norm : str
        Normalization type.
    K : int
        Chunk length.
    num_spks : int
        Number of sources (speakers).
    skip_around_intra : bool
        Skip connection around intra.
    linear_layer_after_inter_intra : bool
        Linear layer after inter and intra.
    use_global_pos_enc : bool
        Global positional encodings.
    max_length : int
        Maximum sequence length.

    Example
    ---------
    >>> intra_block = SBTransformerBlock(1, 64, 8)
    >>> inter_block = SBTransformerBlock(1, 64, 8)
    >>> dual_path_model = Dual_Path_Model(64, 64, intra_block, inter_block, num_spks=2)
    >>> x = torch.randn(10, 64, 2000)
    >>> x = dual_path_model(x)
    >>> x.shape
    torch.Size([2, 10, 64, 2000])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        intra_model: nn.Module,
        inter_model: nn.Module,
        num_layers: int = 1,
        norm: str = "ln",
        K: int = 200,
        num_spks: int = 2,
        skip_around_intra: bool = True,
        linear_layer_after_inter_intra: bool = True,
        use_global_pos_enc: bool = False,
        max_length: int = 20000,
    ):
        super().__init__()

        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = PositionalEncoding(max_length)

        self.dual_mdl = nn.ModuleList([])
        for _ in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    Dual_Computation_Block(
                        intra_model,
                        inter_model,
                        out_channels,
                        norm,
                        skip_around_intra=skip_around_intra,
                        linear_layer_after_inter_intra=linear_layer_after_inter_intra,
                    )
                )
            )

        self.conv2d = nn.Conv2d(out_channels, out_channels * num_spks, kernel_size=1)
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1), nn.Tanh())
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, L].

        Returns
        -------
        out : torch.Tensor
            Output tensor of dimension [spks, B, N, L]
            where, spks = Number of speakers
               B = Batchsize,
               N = number of filters
               L = the number of time points
        """

        # before each line we indicate the shape after executing the line

        # [B, N, L]
        x = self.norm(x)

        # [B, N, L]
        x = self.conv1d(x)
        if self.use_global_pos_enc:
            x = self.pos_enc(x.transpose(1, -1)).transpose(1, -1) + x * (
                x.size(1) ** 0.5
            )

        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)

        # [B, N, K, S]
        for i in range(self.num_layers):
            x = self.dual_mdl[i](x)
        x = self.prelu(x)

        # [B, N*spks, K, S]
        x = self.conv2d(x)
        B, _, K, S = x.shape

        # [B*spks, N, K, S]
        x = x.view(B * self.num_spks, -1, K, S)

        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)

        # [B*spks, N, L]
        x = self.end_conv1x1(x)

        # [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)

        # [spks, B, N, L]
        x = x.transpose(0, 1)

        return x

    def _padding(self, input: torch.Tensor, K: int):
        """Padding the audio times.

        Arguments
        ---------
        K : int
            Chunks of length.
        P : int
            Hop size.
        input : torch.Tensor
            Tensor of size [B, N, L].
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = (
                torch.Tensor(torch.zeros(B, N, gap)).type(input.dtype).to(input.device)
            )
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.dtype).to(input.device)
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input: torch.Tensor, K: int):
        """The segmentation stage splits

        Arguments
        ---------
        K : int
            Length of the chunks.
        input : torch.Tensor
            Tensor with dim [B, N, L].

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points
        """
        B, N, _ = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(B, N, -1, K).transpose(2, 3)

        return input.contiguous(), gap

    def _over_add(self, input: torch.Tensor, gap: int):
        """Merge the sequence with the overlap-and-add method.

        Arguments
        ---------
        input : torch.tensor
            Tensor with dim [B, N, K, S].
        gap : int
            Padding length.

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, L].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points

        """
        B, N, K, _ = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input
