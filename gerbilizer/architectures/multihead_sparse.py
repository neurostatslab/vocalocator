from typing import Optional

import torch
from torch import nn

from .attn_func import sparse_attn


class MultiheadSparseAttn(nn.Module):
    """Implements multi-head attention with a sparse attention span.
    Implementation of BigBird: <https://arxiv.org/abs/2007.14062>.

    I thought about subclassing nn.MultiheadAttention here, but it uses concatenated
    weight matrices for the attention heads, and I'm not sure if that's compatible
    with my implementation of the attention function. To make up for this, most of
    the function signatures will be conserved.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_size: int,
        n_global: int,
        n_window: int,
        n_random: int,
        bias: bool = True,
        batch_first: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(MultiheadSparseAttn, self).__init__()

        param_args = {"device": device, "dtype": dtype}
        self.d_model = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.use_bias = bias
        self.rand_pattern = None

        self.attn_block_size = block_size
        self.attn_n_global = n_global
        self.attn_n_window = n_window
        self.attn_n_random = n_random

        if embed_dim % num_heads != 0:
            raise ValueError(
                "Attention head count must divide model embedding dimension."
            )
        self.d_head = embed_dim // num_heads

        self.q_proj_weight = nn.Parameter(
            torch.empty((num_heads, embed_dim, self.d_head), **param_args)
        )
        self.k_proj_weight = nn.Parameter(
            torch.empty((num_heads, embed_dim, self.d_head), **param_args)
        )
        self.v_proj_weight = nn.Parameter(
            torch.empty((num_heads, embed_dim, self.d_head), **param_args)
        )

        self.register_parameter("q_proj_weight", self.q_proj_weight)
        self.register_parameter("k_proj_weight", self.k_proj_weight)
        self.register_parameter("w_proj_weight", self.v_proj_weight)

        if bias:
            self.q_proj_bias = nn.Parameter(
                torch.empty((num_heads, self.d_head), **param_args)
            )
            self.k_proj_bias = nn.Parameter(
                torch.empty((num_heads, self.d_head), **param_args)
            )
            self.v_proj_bias = nn.Parameter(
                torch.empty((num_heads, self.d_head), **param_args)
            )
            self.register_parameter("q_proj_bias", self.q_proj_bias)
            self.register_parameter("k_proj_bias", self.k_proj_bias)
            self.register_parameter("v_proj_bias", self.v_proj_bias)

        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=bias, **param_args)
        self._init_parameters()

    def _init_parameters(self):
        """Initializes projection weights"""
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)
        if self.use_bias:
            nn.init.constant_(self.q_proj_bias, 0)
            nn.init.constant_(self.k_proj_bias, 0)
            nn.init.constant_(self.v_proj_bias, 0)
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        is_batched = query.dim() == 3  # batch dim + sequence dim + feature dim
        if not self.batch_first and is_batched:
            # Move the batch dim to 0, which my implementation expects
            query, key, value = (a.transpose(0, 1) for a in (query, key, value))

        # Reshape for matmul broadcasting:
        # (batch, seq_len, embed) -> (batch, 1, seq_len, embed)
        query, key, value = (a.unsqueeze(-3) for a in (query, key, value))
        Q = torch.matmul(
            query, self.q_proj_weight
        )  # Has shape (batch, n_head, seq_len, d_head)
        K = torch.matmul(key, self.k_proj_weight)
        V = torch.matmul(value, self.v_proj_weight)
        if self.use_bias:
            if is_batched:
                Q_b = self.q_proj_bias[None, :, None, :]
                K_b = self.k_proj_bias[None, :, None, :]
                V_b = self.v_proj_bias[None, :, None, :]
            else:
                Q_b = self.q_proj_bias[:, None, :]
                K_b = self.k_proj_bias[:, None, :]
                V_b = self.v_proj_bias[:, None, :]
            Q = Q + Q_b
            K = K + K_b
            V = V + V_b

        # Should have shape (batch, n_head, seq_len, d_head)
        attn_out, self.rand_pattern = sparse_attn(
            Q,
            K,
            V,
            block_size=self.attn_block_size,
            n_global=self.attn_n_global,
            n_window=self.attn_n_window,
            n_random=self.attn_n_random,
            rand_idx=self.rand_pattern,
            return_rand_idx=True,
        )

        # Flatten for projection
        # Involves transposing and reshaping to (batch * seq_len, n_head * d_head)

        seq_len = attn_out.shape[-2]
        flat = attn_out.transpose(-2, -3)
        flat = flat.reshape(-1, flat.shape[-1] * flat.shape[-2])
        attn_out = self.out_proj(flat)

        # Reintroduce batch dim
        if is_batched:
            attn_out = attn_out.view(-1, seq_len, self.d_model)
            if not self.batch_first:
                attn_out = attn_out.transpose(0, 1)

        return attn_out
