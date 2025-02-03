import typing as tp

import torch
from torch import nn
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
        w_wrap: torch.Tensor = mod.weight
        b_wrap: tp.Optional[torch.Tensor] = mod.bias
        self.use_bias = b_wrap is not None

        out_size, in_size, k_size = w_wrap.shape
        # Bias shape: (out_size,)

        # Create low rank weight update matrices
        w_lora_A = torch.empty((self.rank, in_size, k_size), dtype=w_wrap.dtype)
        w_lora_B = torch.empty((out_size, self.rank, k_size), dtype=w_wrap.dtype)

        self.register_parameter("w_lora_A", nn.Parameter(w_lora_A))
        self.register_parameter("w_lora_B", nn.Parameter(w_lora_B))
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
        nn.init.zeros_(self.b_lora)

    def _make_kernel(self) -> torch.Tensor:
        """Creates an adapted convolution kernel based on the current low rank weight
        matrices. The kernel is scaled proportionally to the inverse of the module's rank
        to allow for consistency in learning rate across different choices of the rank
        hyperparameter (as done in the paper).
        """

        K = torch.einsum("rik,ork->oik", self.w_lora_A, self.w_lora_B) * self.scale
        K_comb = self.w_wrap + K
        return K_comb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def _make_matrix(self) -> torch.Tensor:
        """Creates an adapted weight matrix based on the current low rank weight
        matrices. The matrix is scaled proportionally to the inverse of the module's rank
        to allow for consistency in learning rate across different choices of the rank
        hyperparameter (as done in the paper).
        """

        W = torch.einsum("ik,kj->ij", self.w_lora_B, self.w_lora_A) * self.scale
        W_comb = self.w_wrap + W
        return W_comb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_comb = (
            self._make_matrix() if self.cached_matrix is None else self.cached_matrix
        )
        if self.use_bias:
            b = self.b_wrap + self.bias
        else:
            b = None
        return F.linear(x, weight=W_comb, bias=b)
