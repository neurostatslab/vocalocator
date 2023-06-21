from copy import copy
from math import comb

import torch
from torch import nn
from torch.nn import functional as F

from .util import build_cov_output
from .simplenet import GerbilizerSimpleLayer

from .encodings import LearnedEncoding, FixedEncoding


class Transpose(nn.Module):
    def __init__(self, dim_a, dim_b):
        super().__init__()
        self.a = dim_a
        self.b = dim_b

    def forward(self, tensor):
        return tensor.transpose(self.a, self.b)


class GerbilizerTransformer(nn.Module):
    """A transformer-based model for the Gerbilizer task."""
    default_config = {
        "CONV_KERNEL_SIZE": 13,

        "NUM_ENCODER_LAYERS": 1,
        "ENCODER_D_MODEL": 512,
        "ENCODER_N_HEADS": 8,
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

        # Define the encoder and decoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_config["ENCODER_D_MODEL"],
            nhead=model_config["ENCODER_N_HEADS"],
            dim_feedforward=model_config["ENCODER_DIM_FEEDFORWARD"],
            dropout=model_config["ENCODER_DROPOUT"],
            activation='relu',
            batch_first=True,
        )
        encoders = [copy(encoder_layer) for _ in range(model_config["NUM_ENCODER_LAYERS"])]

        simple_blocks = [
            nn.Sequential(
                GerbilizerSimpleLayer(
                    n_mics,
                    model_config["ENCODER_D_MODEL"],
                    25,
                    downsample=False,
                    dilation=1,
                ),
                Transpose(1, 2),
            )
        ]
        for _ in range(model_config["NUM_ENCODER_LAYERS"] - 1):
            simple_blocks.append(nn.Sequential(
                Transpose(1, 2),
                GerbilizerSimpleLayer(
                    model_config["ENCODER_D_MODEL"],
                    model_config["ENCODER_D_MODEL"],
                    25,
                    downsample=False,
                    dilation=1,
                ),
                Transpose(1, 2),
            ))

        self.simple_blocks = nn.ModuleList(simple_blocks)
        self.enc_blocks = nn.ModuleList(encoders)
        self.enc_p_encoding = LearnedEncoding(
            d_model=n_mics, max_seq_len=2000,
        )

        # Prediction head
        dense_block = [nn.Linear(model_config["ENCODER_D_MODEL"], model_config["LINEAR_DIM_FEEDFORWARD"]), nn.ReLU()]
        for _ in range(model_config["LINEAR_N_LAYERS"] - 1):
            dense_block.extend([nn.Linear(model_config["LINEAR_DIM_FEEDFORWARD"], model_config["LINEAR_DIM_FEEDFORWARD"]), nn.ReLU()])
        self.dense_block = nn.Sequential(*dense_block)

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
        x = self.enc_p_encoding(x)
        for conv, enc in zip(self.simple_blocks, self.enc_blocks):
            x = conv(x)
            x = enc(x)

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
