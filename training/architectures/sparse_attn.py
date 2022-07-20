from copy import deepcopy
from typing import Callable, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def pad_mat(
    mat: torch.Tensor,
    block_size: int
):
    """ Pads a matrix or batch of matrices (in which the two final dimensions are considered the row
    and column dims of each matrix) such that both dimensions are enlarged to the nearest multiple of
    block_size greater than or equal to their current length.
    
    Params:
        - mat (torch.Tensor): The matrix or batch of matrices to pad
        - block_size (int): Number by which the new dimensions will be divisible
    """
    nearest_mult_d1 = mat.shape[-2] + block_size - (mat.shape[-2] % block_size) if (mat.shape[-2] % block_size != 0) else mat.shape[-2]
    nearest_mult_d2 = mat.shape[-1] + block_size - (mat.shape[-1] % block_size) if (mat.shape[-1] % block_size != 0) else mat.shape[-1]
    padded = torch.zeros((*mat.shape[:-2], nearest_mult_d1, nearest_mult_d2), dtype=mat.dtype, device=mat.device)
    padded[..., :mat.shape[-2], :mat.shape[-1]] = mat
    return padded


def dense_attn(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor, *,
    block_size: int=None,
    mask: torch.Tensor=None
):
    """ Computes dense attention, in which every sequence element may attend to all others, on provided
    query, key, and value sets. Optionally emulates a block-sparse masking protocol.

    Params:
        - q: Query set. Has shape (..., seq_len, d_model)
        - k: Key set. Has shape (..., seq_len, d_model)
        - v: Value set. Has shape (..., seq_len, d_model)
        - block_size: When emulating a block-sparse attention, this specifies the block size to use
        - mask: When emulating a block-sparse attention module, this mask is added to attention weights
        prior to normalization
    """
    orig_len = q.shape[-2]
    scale_factor = np.sqrt(q.shape[-1])
    
    if block_size is not None:
        q, k, v = pad_mat(q, block_size), pad_mat(k, block_size), pad_mat(v, block_size)
    
    scores = torch.matmul(q / scale_factor, k.transpose(-1, -2))
    if mask is not None:
        if orig_len < q.shape[-2]:
            # Simulate the masking of sequence padding done in the sparse implementation
            diff = q.shape[-2] - orig_len
            mask[-diff:, :] = -np.inf
            mask[:, -diff:] = -np.inf
        scores = scores + mask
    scaled_scores = torch.softmax(scores, dim=-1)
    return torch.matmul(scaled_scores, v)[:orig_len, :]


def sparse_attn(q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor, *,
                block_size: int,
                n_global: int,
                n_window: int,
                n_random: int,
                rand_idx: Optional[torch.Tensor]=None,
                return_rand_idx: bool=False,
                mask_value: float=-10000.0
):
    """ Computes sparse attention over the provided keys and queries according to BigBird:
    <https://arxiv.org/abs/2007.14062>.
    Figure 6 in Appendix D is particularly useful in deciphering the math here.
    Expected input shapes:
        - q: (..., seq_len, d_model), float, gpu
        - k: (..., seq_len, d_model), float, gpu
        - v: (..., seq_len, d_model), float, gpu
        block_size: scalar, int
        n_global: scalar, int
        n_window: scalar, odd int
        n_random: scalar, int
    """
    # I think this setup only makes sense for a square attention matrix, but I'm going to
    # separate these two values just in case
    # Saving for the sake of masking padded entries after matrix multiplication
    # Inputs are either (batch, heads, seq_len, d_model) or (batch, seq_len, d_model)
    if q.dim() == 3:
        k, q, v = (a.unsqueeze(1) for a in (k, q, v))  # unsqueeze in attn head dimension
    
    if any(mat.dim() != 4 for mat in (q, k, v)):
        raise ValueError('Query, Key, and Value matrices are expected to have shape (batch, heads, seq_len, d_model) or (batch, seq_len, d_model) or (heads, seq_len, d_model')
    
    batch_size = q.shape[0]
    n_heads = q.shape[1]
    orig_q_len = q.shape[2]
    orig_v_len = v.shape[2]
    d_model = q.shape[3]
    
    if n_global < 0 or n_window < 0 or n_random < 0:
        raise ValueError('Attention parameters n_global, n_window, and n_random should be positive integers')
    
    if n_window % 2 != 1:
        raise ValueError('Attention parameter n_window should be an odd integer')
    
    if orig_q_len != k.shape[2]:
        raise ValueError('Key and Query sequences should have the same length.')
    if q.shape[3] != k.shape[3] or q.shape[3] != v.shape[3]:
        raise ValueError('Query, key, and Value sequence elements should contain the same number of features.')
    
    if orig_q_len % block_size != 0:
        q = pad_mat(q, block_size)
        k = pad_mat(k, block_size)
    if orig_v_len % block_size != 0:
        v = pad_mat(v, block_size)

    q_len = q.shape[2]
    v_len = v.shape[2]

    num_pad = q_len - orig_q_len
    
    # Define some useful values
    n_blocks = q_len // block_size
    
    if n_global + n_window + n_random > n_blocks:
        return dense_attn(q, k, v)
    
    q = q / np.sqrt(d_model)
    # q_toprow: for the tokens that attend to all tokens
    q_toprow, q_sparse = torch.split(q, [n_global * block_size, q_len - n_global * block_size], dim=2)
    q_sparse = q_sparse.view(batch_size, n_heads, -1, block_size, d_model)

    # k_t_toprow = k.transpose(-1, -2)  # unblocked keys for global rows
    
    # k_block_global: blocked keys for the global columns)
    # k_block_sparse: blocked keys for the remaining sparse columns
    k_block_global, k_block_sparse = torch.split(k, [n_global * block_size, q_len - n_global * block_size], dim=2)
    k_block_global = k_block_global.reshape(batch_size, n_heads, -1, block_size, d_model)
    k_block_sparse = k_block_sparse.reshape(batch_size, n_heads, -1, block_size, d_model)
    
    # ==========================================================================
    # Computations for top row of blocks:
    # All keys interact with first n_global queries
    # q (..., n_global*block_size, d) x K^T (..., d, pad_seq_len) -> (..., n_global*block_size, pad_seq_len)
    # top_row_global_scores = torch.matmul(q_toprow, k_t_toprow)
    # Q * K^T
    top_row_attn = torch.einsum('...IJ,...KJ->...IK', q_toprow, k)
    if num_pad > 0:
        # The last `num_pad` columns of the global rows represent zero-pad elements that should not be attended to
        top_row_mask = torch.zeros_like(top_row_attn)
        top_row_mask[..., -num_pad:] = mask_value
        top_row_attn = top_row_attn + top_row_mask

    # ==========================================================================
    # Start with the global columns
    global_col = k_block_global.view(batch_size, n_heads, 1, n_global * block_size, d_model)
    global_col = global_col.expand(-1, -1, n_blocks - n_global, -1, -1)
    global_col_attn = torch.einsum('...IJ,...KJ->...IK', q_sparse, global_col)  # (batch, heads, blocks - n_global, block_size, n_global * block_size)
    global_col_attn = global_col_attn.view(batch_size, n_heads, -1, n_global * block_size)  # (batch, heads, num_nonglobal_elements, num_global_elements)
    if num_pad > 0:
        # The last `num_pad` rows of global_col_attn are zero-padding, should not be attending to anything
        global_col_mask = torch.zeros_like(global_col_attn)
        global_col_mask[..., -num_pad:, :] = mask_value
        global_col_attn = global_col_attn + global_col_mask

    # ==========================================================================
    # Gather the windowed columns
    # Roll out the windows
    win_radius = n_window // 2
    windowed = torch.cat([torch.roll(k_block_sparse, shift, dims=-3) for shift in range(-win_radius, win_radius+1)[::-1]], dim=-2)
    windowed_attn = torch.einsum('...IJ,...KJ->...IK', q_sparse, windowed)
    # windowed_attn will have shape (batch, n_heads, num_nonglobal_elements, n_window * block_size)
    windowed_attn = windowed_attn.view(batch_size, n_heads, q_len - block_size * n_global, n_window * block_size)

    # Attempt to mask the padded elements
    if num_pad > 0:
        windowed_attn_mask = torch.zeros_like(windowed_attn)
        # Prevent the padded elements from attending to any other values
        windowed_attn_mask[..., -num_pad:, :] = mask_value
        # Now prevent other values from attending to the padded elements
        # The last `win_radius + 1` rows will include the final block in their receptive field
        ng_blocks = n_blocks - n_global  # num non-global blocks
        for row_idx in range(ng_blocks - win_radius - 1, ng_blocks):
            # On row r (indexed such that 0 is the first non-global row), ng_blocks - r - 1 represents the index
            # (relative to the window center) of the block containing the final elements of the sequence
            block_idx = ng_blocks - row_idx - 1
            win_center = win_radius * block_size
            illegal_start_idx = win_center + block_idx * block_size + (block_size - num_pad)
            # Everything after illegal_start_idx should be excluded, since they will either be pad elements
            # or blocks that wrapped around to the start of the sequence
            windowed_attn_mask[..., row_idx*block_size:(row_idx+1)*block_size, illegal_start_idx:] = mask_value
        
        # The first `win_radius` rows will attempt to attend to the end of the sequence
        # , since the windows wrap around
        for row_idx in range(win_radius):
            num_illegal_blocks = win_radius - row_idx
            good_start_idx = num_illegal_blocks * block_size  # Everything after this point is OK
            windowed_attn_mask[..., row_idx*block_size:(row_idx+1)*block_size, :good_start_idx] = mask_value
        windowed_attn = windowed_attn + windowed_attn_mask
    
    # ==========================================================================
    # Gather the random columns
    if rand_idx is None:
        # Computed s.t. there is no intersection between the random blocks and the window/global blocks
        win_start = lambda row: row - win_radius  # index of the first key block in the sliding window
        win_end = lambda row: row + win_radius  # index of the last key block in the window
        # Valid indices from which random blocks may be selected given a row (relative to the end of the global rows)
        rand_valid_idx = lambda row: np.array([a for a in range(n_blocks - n_global) if a < win_start(row) or a > win_end(row)])

        # rewriting this to use torch rand functions instead of numpy.random.choice to ensure it works well
        # with torch's checkpointing
        rand_sampled_cols = []
        rand_requires_masking = []
        for row in range(n_blocks-n_global):
            valid_indices = rand_valid_idx(row)
            rand_selection = torch.randint(0, len(valid_indices), (n_random,)).numpy()
            rand_selection.sort()  # makes masking easier
            rand_requires_masking.append(rand_selection[-1] == n_blocks-n_global-1)
            rand_sampled_cols.append(valid_indices[rand_selection])
        
        # Items in dim 2 will be ordered such that we can move n_random to the next dim while remaining contiguous
        rand_gather_idx = torch.zeros((1, 1, k_block_sparse.shape[-3] * n_random, 1, 1), device=k_block_sparse.device, dtype=torch.int64)
        for row, valid_idx in enumerate(rand_sampled_cols):
            rand_gather_idx[0, 0, row * n_random:(row + 1) * n_random, 0, 0] = torch.from_numpy(valid_idx)
        # rand_gather_idx = rand_gather_idx.expand((1, n_heads, -1, block_size, d_model))
    else:
        rand_gather_idx = rand_idx
        rand_requires_masking = [rand_gather_idx[0, 0, (row+1)*n_random-1, 0, 0] == (n_blocks - n_global - 1) for row in range(n_blocks-n_global)]
        
    # For cached rand_gather_idx, the expansion to batch_size prevents the validation batches, which tend to be of different size
    # than train batches or sized inconsistently, from causing problems
    rand_cols = torch.gather(k_block_sparse, dim=2, index=rand_gather_idx.expand((batch_size, n_heads, -1, block_size, d_model)))
    rand_cols = rand_cols.view(batch_size, n_heads, k_block_sparse.shape[-3], n_random * block_size, d_model)
    rand_attn = torch.einsum('...IJ,...KJ->...IK', q_sparse, rand_cols)  # Should have shape (batch_size, n_heads, non_global_blocks, block_size, n_random*block_size)

    if num_pad > 0:
        rand_attn_mask = torch.zeros_like(rand_attn)
        for row, needs_mask in enumerate(rand_requires_masking):
            if not needs_mask:
                continue
            # Prevent zero-pad elements from being attended to randomly
            # Since the random indices were sorted, the zero-pad elements are located at the end of the last dim
            rand_attn_mask[:, :, row, :, -num_pad:] = mask_value
        # Prevent zero-pad elements from randomly attending to anything
        rand_attn_mask[:, :, -1, -num_pad:, :] = mask_value
        rand_attn = rand_attn + rand_attn_mask
    # I need all the attention weight tensors to have the same shape, reshaping this to (batch, n_heads, n_nonglobal_elements, n_rand * block_size)
    rand_attn = rand_attn.view(batch_size, n_heads, -1, n_random * block_size)

    # ==========================================================================
    # Softmax
    # Goal here is to perform a softmax without actually concatenating the involved tensors
    # concatenation is expensive so I aim to do it only once at the very end
    # reminder to myself that the attn variable names are: top_row_attn, global_col_attn, windowed_attn, rand_attn
    top_row_scores = torch.softmax(top_row_attn, dim=-1)  # These contain every column already, can use normal softmax

    gcol_max = global_col_attn.max(dim=-1, keepdims=True)[0]
    wcol_max = windowed_attn.max(dim=-1, keepdims=True)[0]
    rcol_max = rand_attn.max(dim=-1, keepdims=True)[0]  # these three are temporary

    # Subtracting the maximum is just for numerical stability. Softmax is invariant to translation
    overall_max = torch.maximum(torch.maximum(gcol_max, wcol_max), rcol_max)

    # Compute numerators
    global_col_scores = torch.exp(global_col_attn - overall_max)
    windowed_scores = torch.exp(windowed_attn - overall_max)
    rand_scores = torch.exp(rand_attn - overall_max)

    # Calc. denominator
    denominator = global_col_scores.sum(dim=-1, keepdims=True) + windowed_scores.sum(dim=-1, keepdims=True) + rand_scores.sum(dim=-1, keepdims=True)

    global_col_scores = global_col_scores / denominator
    windowed_scores = windowed_scores / denominator
    rand_scores = rand_scores / denominator

    # ==========================================================================
    # Collect necessary value columns
    top_row_values = v  # Pretty straightforward here, they attend to all values
    top_row_output = torch.einsum('...IJ,...JK->...IK', top_row_scores, top_row_values)

    global_col_vals, v_sparse = torch.split(v, [n_global * block_size, v_len - n_global * block_size], dim=2)
    # (batch, heads, n_nonglobal_elements, n_global_elements) x (batch, heads, n_global_elements, d_model)
    global_col_output = torch.einsum('...IJ,...JK->...IK', global_col_attn, global_col_vals)

    # Value indexing pretty much mirrors key indexing
    v_block_sparse = v_sparse.view(batch_size, n_heads, -1, block_size, d_model)
    windowed_values = torch.cat([torch.roll(v_block_sparse, shift, dims=-3) for shift in range(-win_radius, win_radius+1)[::-1]], dim=-2)
    # Reshape from (batch_size, n_heads, n_nonglobal_elements, n_window_elements) -> (batch_size, n_heads, n_nonglobal_blocks, block_size, n_window_elements)
    windowed_scores = windowed_scores.view(batch_size, n_heads, -1, block_size, block_size * n_window)
    # Each score needs to be broadcast out 'block_size' times, since they all share the same value window
    windowed_output = torch.einsum('...IJ,...JK->...IK', windowed_scores, windowed_values).view(batch_size, n_heads, -1, d_model)


    # Random columns
    v_rand = torch.gather(v_block_sparse, dim=2, index=rand_gather_idx.expand((batch_size, n_heads, -1, block_size, d_model)))
    v_rand = v_rand.view(batch_size, n_heads, v_block_sparse.shape[-3], n_random * block_size, d_model)
    # Same thing that happened with the windows happens here
    rand_scores = rand_scores.view(batch_size, n_heads, -1, block_size, n_random * block_size)
    rand_output = torch.einsum('...IJ,...JK->...IK', rand_scores, v_rand).view(batch_size, n_heads, -1, d_model)
    
    # ==========================================================================
    # Combine all the different attention patterns
    pad_sparse_output = global_col_output + windowed_output + rand_output
    sparse_output = pad_sparse_output if num_pad == 0 else pad_sparse_output[..., :-num_pad, :]
    final_output = torch.cat([top_row_output, sparse_output], dim=-2)

    if return_rand_idx:
        return final_output, rand_gather_idx
    return final_output


class MultiheadSparseAttn(nn.Module):
    """ Implements multi-head attention with a sparse attention span.
    Implementation of BigBird: <https://arxiv.org/abs/2007.14062>.
    
    I thought about subclassing nn.MultiheadAttention here, but it uses concatenated
    weight matrices for the attention heads, and I'm not sure if that's compatible
    with my implementation of the attention function. To make up for this, most of
    the function signatures will be conserved.
    """
    def __init__(self,
        embed_dim: int,
        num_heads: int,
        block_size: int,
        n_global: int,
        n_window: int,
        n_random: int,
        bias: bool=True,
        batch_first: bool=False,
        device: Optional[torch.device]=None,
        dtype: Optional[torch.dtype]=None
    ):
        super(MultiheadSparseAttn, self).__init__()

        param_args = {'device': device, 'dtype': dtype}
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
            raise ValueError('Attention head count must divide model embedding dimension.')
        self.d_head = embed_dim // num_heads

        self.q_proj_weight = nn.Parameter(torch.empty((num_heads, embed_dim, self.d_head), **param_args))
        self.k_proj_weight = nn.Parameter(torch.empty((num_heads, embed_dim, self.d_head), **param_args))
        self.v_proj_weight = nn.Parameter(torch.empty((num_heads, embed_dim, self.d_head), **param_args))
        
        self.register_parameter('q_proj_weight', self.q_proj_weight)
        self.register_parameter('k_proj_weight', self.k_proj_weight)
        self.register_parameter('w_proj_weight', self.v_proj_weight)

        if bias:
            self.q_proj_bias = nn.Parameter(torch.empty((num_heads, self.d_head), **param_args))
            self.k_proj_bias = nn.Parameter(torch.empty((num_heads, self.d_head), **param_args))
            self.v_proj_bias = nn.Parameter(torch.empty((num_heads, self.d_head), **param_args))
            self.register_parameter('q_proj_bias', self.q_proj_bias)
            self.register_parameter('k_proj_bias', self.k_proj_bias)
            self.register_parameter('v_proj_bias', self.v_proj_bias)
        
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=bias, **param_args)
        self._init_parameters()
    
    def _init_parameters(self):
        """ Initializes projection weights
        """
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)
        if self.use_bias:
            nn.init.constant_(self.q_proj_bias, 0)
            nn.init.constant_(self.k_proj_bias, 0)
            nn.init.constant_(self.v_proj_bias, 0)
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, 
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ):
        is_batched = query.dim() == 3  # batch dim + sequence dim + feature dim
        if not self.batch_first and is_batched:
            # Move the batch dim to 0, which my implementation expects
            query, key, value = (a.transpose(0, 1) for a in (query, key, value))
        
        # Reshape for matmul broadcasting:
        # (batch, seq_len, embed) -> (batch, 1, seq_len, embed)
        query, key, value = (a.unsqueeze(-3) for a in (query, key, value))
        Q = torch.matmul(query, self.q_proj_weight)  # Has shape (batch, n_head, seq_len, d_head)
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
            Q, K, V,
            block_size = self.attn_block_size,
            n_global = self.attn_n_global,
            n_window = self.attn_n_window,
            n_random = self.attn_n_random,
            rand_idx = self.rand_pattern,
            return_rand_idx = True
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


class SparseTransformerEncoderLayer(nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        block_size: int,
        n_global: int,
        n_window: int,
        n_random: int,
        dim_feedforward: int,
        dropout: float=0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        batch_first: bool=True,
        norm_first: bool=False,
        device: Optional[torch.device]=None,
        dtype: Optional[torch.dtype]=None
    ):
        super(SparseTransformerEncoderLayer, self).__init__()

        self.d_model = d_model
        self.n_head = num_heads
        self.batch_first = batch_first
        self.norm_first = norm_first

        param_args = {'device': device, 'dtype': dtype}

        if isinstance(activation, str):
            if activation.lower() == 'relu':
                activation = F.relu
            elif activation.lower() == 'gelu':
                activation = F.gelu
            else:
                raise ValueError(f'Unsupported activation function: {activation}')
        self.activation = activation

        self.attn = MultiheadSparseAttn(
            embed_dim=d_model,
            num_heads=num_heads,
            block_size=block_size,
            n_global=n_global,
            n_window=n_window,
            n_random=n_random,
            bias=True,
            batch_first=batch_first,
            **param_args
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
        x = self.attn(x, x, x)
        return self.dropout1(x)


class SparseTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, n_layers):
        super(SparseTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([deepcopy(encoder_layer) for _ in range(n_layers)])
    
    def forward(self, x):
        transformer_out = x
        for layer in self.layers:
            transformer_out = layer(transformer_out)
        return transformer_out