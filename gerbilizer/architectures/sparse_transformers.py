from copy import deepcopy
from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .multihead_sparse import MultiheadSparseAttn


class LearnedEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        enc = torch.empty((max_seq_len, d_model))
        nn.init.uniform_(enc, -0.1, 0.1)
        self.encoding = nn.Parameter(enc, requires_grad=True)

    def forward(self, x):
        return x + self.encoding[: x.shape[1], :].unsqueeze(0)


class SparseTransformerEncoderLayer(nn.Module):
    """Mirrors PyTorch's implementation of TransformerEncoderLayer,
    but replaces nn.MultiHeadSparseAttention with my own class.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        block_size: int,
        n_global: int,
        n_window: int,
        n_random: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        checkpoint: bool = False,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        batch_first: bool = True,
        norm_first: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(SparseTransformerEncoderLayer, self).__init__()

        self.d_model = d_model
        self.n_head = num_heads
        self.batch_first = batch_first
        self.norm_first = norm_first

        param_args = {"device": device, "dtype": dtype}

        if isinstance(activation, str):
            if activation.lower() == "relu":
                activation = F.relu
            elif activation.lower() == "gelu":
                activation = F.gelu
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
        self.activation = activation
        self.checkpoint = checkpoint

        self.attn = MultiheadSparseAttn(
            embed_dim=d_model,
            num_heads=num_heads,
            block_size=block_size,
            n_global=n_global,
            n_window=n_window,
            n_random=n_random,
            bias=True,
            batch_first=batch_first,
            **param_args,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward, **param_args)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **param_args)

        self.dropout = nn.Dropout(dropout)  # dropout inside residual black
        self.dropout1 = nn.Dropout(dropout)  # dropout on self-attention result
        self.dropout2 = nn.Dropout(dropout)  # another dropout inside residual block

        self.norm1 = nn.LayerNorm(d_model, **param_args)
        self.norm2 = nn.LayerNorm(d_model, **param_args)

    def forward(self, x):
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def _sa_block(self, x):
        if self.checkpoint:
            x = torch_checkpoint(self.attn, x, x, x, preserve_rng_state=True)
        else:
            x = self.attn(x, x, x)
        return self.dropout1(x)


class SparseTransformerEncoder(nn.Module):
    """Also mirror's PyTorch's implementation of TransformerEncoder."""

    def __init__(self, encoder_layer, n_layers):
        super(SparseTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([deepcopy(encoder_layer) for _ in range(n_layers)])

    def forward(self, x):
        transformer_out = x
        for layer in self.layers:
            transformer_out = layer(transformer_out)
        return transformer_out
