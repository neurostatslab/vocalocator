from copy import copy
from math import comb
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .util import build_cov_output
from .simplenet import GerbilizerSimpleLayer

from .encodings import LearnedEncoding
from .rope import RotaryPEMultiHeadSelfAttention


class Transpose(nn.Module):
    def __init__(self, dim_a, dim_b):
        super().__init__()
        self.a = dim_a
        self.b = dim_b

    def forward(self, tensor):
        return tensor.transpose(self.a, self.b)
    

def swish(x):
    return x * torch.sigmoid(x)  # Swish activation function with beta=1

class SwiGLUEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            layer_norm_eps: float = 1e-5,
            batch_first: bool = False,
            norm_first: bool = False,
            device: Optional[torch.device]=None,
            dtype: Optional[torch.dtype]=None
    ) -> None:
        super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                        dropout=dropout, batch_first=batch_first, norm_first=norm_first,
                        layer_norm_eps=layer_norm_eps, device=device, dtype=dtype)

        self.batch_first = batch_first
        # Remove bias from the feedforward layer
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        # overwrite the feedforward layer with a SwiGLU layer
        self.linearV = nn.Linear(d_model, dim_feedforward, bias=False)

        # Positional encoding returns data with batch second
        # self.self_attn = RotaryPEMultiHeadSelfAttention(d_model, nhead, batch_first=False, device=device, dtype=dtype)
    
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        swished = swish(self.linear1(x))
        prod = self.linearV(x)
        return self.dropout2(self.linear2(self.dropout1(swished * prod)))


class GerbilizerTransformer(nn.Module):
    """A transformer-based model for the Gerbilizer task."""
    default_config = {
        "CONV_KERNEL_SIZE": 51,
        "CONV_N_LAYERS": 3,

        "NUM_ENCODER_LAYERS": 5,
        "ENCODER_D_MODEL": 256,
        "ENCODER_N_HEADS": 32,
        "ENCODER_DROPOUT": 0.1,
        "ENCODER_DIM_FEEDFORWARD": 2048,

        "LINEAR_N_LAYERS": 3,  # For prediction head
        "LINEAR_DIM_FEEDFORWARD": 2048,
    }

    def __init__(self, config):
        super().__init__()

        n_mics = config["DATA"]["NUM_MICROPHONES"]
        if config["DATA"]["COMPUTE_XCORRS"]:
            n_mics += comb(n_mics, 2)

        # Grab model parameters and fill in missing values with defaults
        model_config = GerbilizerTransformer.default_config.copy()
        model_config.update(config.get("MODEL_PARAMS", {}))
        

        # Convolutional backbone
        conv_kernel_size = model_config["CONV_KERNEL_SIZE"]
        # conv_stride = max(1, conv_kernel_size // 2)
        conv_stride = 2
        c_0 = nn.Conv1d(n_mics, model_config["ENCODER_D_MODEL"], conv_kernel_size, stride=conv_stride, padding='same' if conv_stride == 1 else 0)
        c = nn.Conv1d(model_config["ENCODER_D_MODEL"], model_config["ENCODER_D_MODEL"], conv_kernel_size, stride=conv_stride, padding='same' if conv_stride == 1 else 0)
        conv_layers = [c_0, nn.ReLU()]
        for _ in range(model_config["CONV_N_LAYERS"] - 1):
            conv_layers += [copy(c)]
            conv_layers += [nn.ReLU()]

        self.conv = nn.Sequential(
            Transpose(1, 2),
            *conv_layers,
            Transpose(1, 2),
        )

        # Define the encoder and decoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_config["ENCODER_D_MODEL"],
            nhead=model_config["ENCODER_N_HEADS"],
            dim_feedforward=model_config["ENCODER_DIM_FEEDFORWARD"],
            dropout=model_config["ENCODER_DROPOUT"],
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_config["NUM_ENCODER_LAYERS"],
        )

        # Prediction head
        dense_block = [nn.Linear(model_config["ENCODER_D_MODEL"], model_config["LINEAR_DIM_FEEDFORWARD"]), nn.ReLU()]
        for _ in range(model_config["LINEAR_N_LAYERS"] - 1):
            dense_block.extend([nn.Linear(model_config["LINEAR_DIM_FEEDFORWARD"], model_config["LINEAR_DIM_FEEDFORWARD"]), nn.ReLU()])
        self.dense_block = nn.Sequential(*dense_block)
        self.p_encoding = LearnedEncoding(d_model=model_config["ENCODER_D_MODEL"], max_seq_len=10000)

        self.output_cov = model_config.get("OUTPUT_COV", False)
        n_outputs = 5 if self.output_cov else 2
        self.prob_readout = nn.Linear(model_config["LINEAR_DIM_FEEDFORWARD"], n_outputs)


    def clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)

    def embed(self, x):
        """I'm splitting the forward pass into this 'embedding' segment
        and the final x-y coord output to make inheritance easier
        """
        # x initial shape: (batch_size, seq_ln, n_mics)
        x = self.conv(x)
        cls_token = torch.zeros([x.shape[0], 1, x.shape[2]], device=x.device)
        x = torch.cat([cls_token, x], dim=1)
        x = self.p_encoding(x)
        x = self.encoder(x)

        return self.dense_block(x[:, 0, :])

    def forward(self, x):
        linear_out = self.embed(x)

        readout = self.prob_readout(linear_out)
        if self.output_cov:
            return build_cov_output(readout, readout.device)
        return readout
    
    def trainable_params(self):
        """Useful for freezing parts of the model"""
        return self.parameters()
