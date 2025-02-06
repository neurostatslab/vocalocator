import typing as tp

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LORA_Conv1d(nn.Module):
    def __init__(self, mod: nn.Conv1d, rank: int):
        super(LORA_Conv1d, self).__init__()
        self.rank = rank
        self.scale = 1 / rank

        # Copy other conv parameters from the wrapped obj
        self.dilation = mod.dilation
        self.stride = mod.stride
        self.padding = mod.padding
        if mod.groups != 1:
            raise NotImplementedError("LORA not implemented for groups > 1")

        # Get weights
        w_wrap: Tensor = mod.weight
        b_wrap: tp.Optional[Tensor] = mod.bias
        self.use_bias = b_wrap is not None

        out_size, in_size, k_size = w_wrap.shape
        # Bias shape: (out_size,)

        # Create low rank weight update matrices
        # Tensor outer product
        w_lora_A = torch.empty((self.rank, out_size), dtype=w_wrap.dtype)
        w_lora_B = torch.empty((self.rank, in_size), dtype=w_wrap.dtype)
        w_lora_C = torch.empty((self.rank, k_size), dtype=w_wrap.dtype)

        self.register_parameter("w_lora_A", nn.Parameter(w_lora_A))
        self.register_parameter("w_lora_B", nn.Parameter(w_lora_B))
        self.register_parameter("w_lora_C", nn.Parameter(w_lora_C))
        if self.use_bias:
            b_lora = torch.zeros_like(b_wrap)
            self.register_parameter("b_lora", nn.Parameter(b_lora))

        # Keep a cached version of the adapted kernel to use in inference
        self.register_buffer("cached_kernel", None)

        self.register_buffer("w_wrap", w_wrap, persistent=True)
        if self.use_bias:
            self.register_buffer("b_wrap", b_wrap, persistent=True)

        self.initialize_params()
        self.to(mod.weight.device)

    def train(self, mode: bool = True) -> nn.Module:
        """If entering eval mode, creates a cached kernel which will be reused
        across evaluations of the wrapper.
        If entering train mode, deletes the cached kernel.

        Returns self
        """
        if mode:
            self.cached_kernel = None
        else:
            self.cached_kernel = self._make_kernel()
        return super().train(mode)

    def eval(self) -> nn.Module:
        self.train(False)

    def initialize_params(self):
        nn.init.xavier_normal_(self.w_lora_A)
        nn.init.zeros_(self.w_lora_B)
        nn.init.xavier_normal_(self.w_lora_C)
        nn.init.zeros_(self.b_lora)

    def _make_kernel(self) -> Tensor:
        """Creates an adapted convolution kernel based on the current low rank weight
        matrices. The kernel is scaled proportionally to the inverse of the module's rank
        to allow for consistency in learning rate across different choices of the rank
        hyperparameter (as done in the paper).
        """

        # goal shape: (out, in, kernel)
        K = (
            torch.einsum("ro,ri,rk->oik", self.w_lora_A, self.w_lora_B, self.w_lora_C)
            * self.scale
        )
        K_comb = self.w_wrap + K
        return K_comb

    def forward(self, x: Tensor) -> Tensor:
        K_comb = (
            self._make_kernel() if self.cached_kernel is None else self.cached_kernel
        )
        if self.use_bias:
            b = self.b_wrap + self.b_lora
        else:
            b = None
        return F.conv1d(
            x,
            weight=K_comb,
            bias=b,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
        )


class LORA_Linear(nn.Module):
    def __init__(self, mod: nn.Module, rank: int):
        super(LORA_Linear, self).__init__()
        self.wrapped = mod
        self.rank = rank
        self.scale = 1 / rank

        # Copy other linear parameters from the wrapped obj
        out_size, in_size = mod.weight.shape
        if mod.bias is not None:
            self.use_bias = True
            self.register_parameter("bias", nn.Parameter(torch.zeros(out_size)))
        else:
            self.use_bias = False

        # Create low rank weight update matrices
        w_lora_A = torch.empty((self.rank, in_size), dtype=mod.weight.dtype)
        w_lora_B = torch.empty((out_size, self.rank), dtype=mod.weight.dtype)

        self.register_parameter("w_lora_A", nn.Parameter(w_lora_A))
        self.register_parameter("w_lora_B", nn.Parameter(w_lora_B))

        # Keep a cached version of the adapted weight matrix to use in inference
        self.register_buffer("cached_matrix", None)

        # Cache the weights of the wrapped module as buffers so
        # they are not seen by the optimizer
        self.register_buffer("w_wrap", mod.weight, persistent=True)
        if self.use_bias:
            self.register_buffer("b_wrap", mod.bias, persistent=True)

        self.initialize_params()
        self.to(mod.weight.device)

    def train(self, mode: bool = True) -> nn.Module:
        """If entering eval mode, creates a cached matrix which will be reused
        across evaluations of the wrapper.
        If entering train mode, deletes the cached matrix.

        Returns self
        """

        if mode:
            self.cached_matrix = None
        else:
            self.cached_matrix = self._make_matrix()
        return super().train(mode)

    def eval(self) -> nn.Module:
        self.train(False)

    def initialize_params(self):
        nn.init.xavier_normal_(self.w_lora_A)
        nn.init.zeros_(self.w_lora_B)

    def _make_matrix(self) -> Tensor:
        """Creates an adapted weight matrix based on the current low rank weight
        matrices. The matrix is scaled proportionally to the inverse of the module's rank
        to allow for consistency in learning rate across different choices of the rank
        hyperparameter (as done in the paper).
        """

        W = torch.einsum("ik,kj->ij", self.w_lora_B, self.w_lora_A) * self.scale
        W_comb = self.w_wrap + W
        return W_comb

    def forward(self, x: Tensor) -> Tensor:
        W_comb = (
            self._make_matrix() if self.cached_matrix is None else self.cached_matrix
        )
        if self.use_bias:
            b = self.b_wrap + self.bias
        else:
            b = None
        return F.linear(x, weight=W_comb, bias=b)


class LORA_MHA(nn.Module):
    """A wrapper for torch multihead attention modules which applies low-rank weight
    updates to the QKV matrices. The rank hyperparameter controls the rank of the matrix
    decomposition
    """

    def __init__(self, mod: nn.MultiheadAttention, rank: int):
        super(LORA_MHA, self).__init__()

        self.rank = rank
        self.embed_dim = mod.embed_dim
        self.num_heads = mod.num_heads
        self.dropout = mod.dropout
        self.add_zero_attn = mod.add_zero_attn
        self.batch_first = mod.batch_first
        self.dtype = None

        # Make buffers for the original QKV matrices
        if mod.in_proj_weight is None:
            # module loaded using qkv_same_embed_dim = False
            self.register_buffer("Q", mod.q_proj_weight, persistent=True)
            self.register_buffer("K", mod.k_proj_weight, persistent=True)
            self.register_buffer("V", mod.v_proj_weight, persistent=True)
            self.register_buffer("in_proj_weight", None, persistent=True)
            self.dtype = mod.q_proj_weight.dtype
        else:
            # q, k, v = torch.chunk(mod.in_proj_weight, 3)
            self.register_buffer("Q", None, persistent=True)
            self.register_buffer("K", None, persistent=True)
            self.register_buffer("V", None, persistent=True)
            self.register_buffer("in_proj_weight", mod.in_proj_weight, persistent=True)
            self.dtype = mod.in_proj_weight.dtype

        # Save these because we aren't gonna mess with them
        self.register_buffer("in_proj_bias", mod.in_proj_bias, persistent=True)
        self.register_buffer("bias_k", mod.bias_k, persistent=True)
        self.register_buffer("bias_v", mod.bias_v, persistent=True)
        self.register_buffer("out_proj_weight", mod.out_proj.weight, persistent=True)
        self.register_buffer("out_proj_bias", mod.out_proj.bias, persistent=True)

        # Create low rank weight update matrices
        # These must be separate because keeping them together forces
        # the rhs of the decomposition to be the same for all three

        if self.Q is not None:
            q_out_size, q_in_size = self.Q.shape
            k_out_size, k_in_size = self.K.shape
            # v_out_size, v_in_size = self.V.shape
        else:
            out_size, in_size = self.in_proj_weight.shape
            # q_out_size = k_out_size = v_out_size = out_size // 3
            # q_in_size = k_in_size = v_in_size = in_size
            q_out_size = k_out_size = out_size // 3
            q_in_size = k_in_size = in_size

        q_lora_A = torch.empty((self.rank, q_in_size), dtype=self.dtype)
        q_lora_B = torch.empty((q_out_size, self.rank), dtype=self.dtype)
        k_lora_A = torch.empty((self.rank, k_in_size), dtype=self.dtype)
        k_lora_B = torch.empty((k_out_size, self.rank), dtype=self.dtype)
        # The original LoRA paper finds that updating the V matrix doesn't yield significant benefits
        # v_lora_A = torch.empty((self.rank, v_in_size), dtype=self.V.dtype)
        # v_lora_B = torch.empty((v_out_size, self.rank), dtype=self.V.dtype)

        self.register_parameter("q_lora_A", nn.Parameter(q_lora_A))
        self.register_parameter("q_lora_B", nn.Parameter(q_lora_B))
        self.register_parameter("k_lora_A", nn.Parameter(k_lora_A))
        self.register_parameter("k_lora_B", nn.Parameter(k_lora_B))

        self.register_buffer("cached_Q", None, persistent=False)
        self.register_buffer("cached_K", None, persistent=False)

        self.initialize_params()

    def initialize_params(self):
        nn.init.xavier_normal_(self.q_lora_A)
        nn.init.zeros_(self.q_lora_B)
        nn.init.xavier_normal_(self.k_lora_A)
        nn.init.zeros_(self.k_lora_B)

    def _make_QK(self) -> tp.Tuple[Tensor, Tensor]:
        """Creates an adapted Q and K matrix based on the current low rank weight
        matrices. The matrices are scaled proportionally to the inverse of the module's rank
        to allow for consistency in learning rate across different choices of the rank
        hyperparameter (as done in the paper).
        """

        if self.Q is None:
            Q, K, _ = torch.chunk(self.in_proj_weight, 3, dim=0)
        else:
            Q, K = self.Q, self.K

        # Q = BA + Q_old
        Q = torch.addmm(Q, self.q_lora_B, self.q_lora_A)
        K = torch.addmm(K, self.k_lora_B, self.k_lora_A)
        return Q, K

    def train(self, mode: bool = True) -> nn.Module:
        """If entering eval mode, creates cached Q and K matrices which will be reused
        across evaluations of the wrapper.
        If entering train mode, deletes the cached matrices.

        Args:
            mode (bool, optional): Whether the module is entering train mode. Defaults to True.

        Returns:
            nn.Module: self
        """

        if mode:
            self.cached_Q = None
            self.cached_K = None
        else:
            self.cached_Q, self.cached_K = self._make_QK()
        return super().train(mode)

    def eval(self) -> nn.Module:
        """See train(). Equivalent to train(False)

        Returns:
            nn.Module: self
        """
        self.train(False)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: tp.Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: tp.Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        """Modifies the weight matrices and makes F.mhaf do all the heavy lifting"""

        is_batched = query.dim() == 3

        if self.cached_Q is not None:
            Q = self.cached_Q
            K = self.cached_K
        else:
            Q, K = self._make_QK()

        if self.V is None:
            _, _, V = torch.chunk(self.in_proj_weight, 3, dim=0)
        else:
            V = self.V

        if is_batched and self.batch_first:
            Q = Q.transpose(0, 1)
            K = K.transpose(0, 1)
            V = V.transpose(0, 1)

        attn_output, attn_weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            in_proj_weight=None,
            in_proj_bias=self.in_proj_bias,
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=self.out_proj_weight,
            out_proj_bias=self.out_proj_bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=Q,
            k_proj_weight=K,
            v_proj_weight=V,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        if is_batched and self.batch_first:
            attn_output = attn_output.transpose(0, 1)
        return attn_output, attn_weights
