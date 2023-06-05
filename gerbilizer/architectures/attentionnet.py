from itertools import chain
from math import comb

import torch
from torch import nn
from torch.nn import functional as F

from gerbilizer.architectures.util import build_cov_output

from .encodings import LearnedEncoding, FixedEncoding


class Transpose(nn.Module):
    def __init__(self, dim_a, dim_b):
        super().__init__()
        self.a = dim_a
        self.b = dim_b

    def forward(self, tensor):
        return tensor.transpose(self.a, self.b)


class Skip(nn.Module):
    def __init__(self, submodule):
        super().__init__()
        self.submodule = submodule

    def forward(self, x):
        return x + self.submodule(x)


class GerbilizerTransformer(nn.Module):
    """A transformer-based model for the Gerbilizer task."""
    default_config = {
        "CONV_KERNEL_SIZE": 13,
        "CONV_STRIDE": 2,
        "CONV_DILATION": 2,
        "CONV_N_LAYERS": 1,

        "NUM_ENCODER_LAYERS": 1,
        "ENCODER_D_MODEL": 512,
        "ENCODER_N_HEADS": 8,
        "ENCODER_DROPOUT": 0.1,
        "ENCODER_DIM_FEEDFORWARD": 2048,
        "DECODER_D_MODEL": 512,
        "DECODER_N_HEADS": 8,
        "DECODER_DROPOUT": 0.1,
        "DECODER_DIM_FEEDFORWARD": 2048,

        "OUTPUT_WIDTH": 40,
        "OUTPUT_HEIGHT": 30,

        "LINEAR_N_LAYERS": 1,  # For prediction head
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
        conv_n_channels = [n_mics] + model_config["CONV_N_LAYERS"] * [model_config["ENCODER_D_MODEL"]]
        conv_kernel_size = model_config["CONV_KERNEL_SIZE"]
        conv_stride = model_config["CONV_STRIDE"]
        conv_dilation = model_config["CONV_DILATION"]

        self.conv_layers = nn.ModuleList()
        for in_channels, out_channels in zip(conv_n_channels[:-1], conv_n_channels[1:]):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=conv_kernel_size,
                stride=conv_stride,
                dilation=conv_dilation,
            )
            if not self.conv_layers:
                self.conv_layers.append(conv)
            else:  # Can't add a skip connection to the first layer because n_mics << d_model
                self.conv_layers.append(Skip(conv))
            self.conv_layers.append(nn.ReLU())

        # Define the encoder and decoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_config["ENCODER_D_MODEL"],
            nhead=model_config["ENCODER_N_HEADS"],
            dim_feedforward=model_config["ENCODER_DIM_FEEDFORWARD"],
            dropout=model_config["ENCODER_DROPOUT"],
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_config["NUM_ENCODER_LAYERS"],
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_config["DECODER_D_MODEL"],
            nhead=model_config["DECODER_N_HEADS"],
            dim_feedforward=model_config["DECODER_DIM_FEEDFORWARD"],
            dropout=model_config["DECODER_DROPOUT"],
            activation='gelu',
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=1,
        )

        self.enc_p_encoding = LearnedEncoding(
            d_model=conv_n_channels[-1], max_seq_len=2000,
        )

        # Prediction head
        dense_block = [nn.Linear(model_config["DECODER_D_MODEL"], model_config["LINEAR_DIM_FEEDFORWARD"]), nn.ReLU()]
        for _ in range(model_config["LINEAR_N_LAYERS"] - 1):
            dense_block.extend([nn.Linear(model_config["LINEAR_DIM_FEEDFORWARD"], model_config["LINEAR_DIM_FEEDFORWARD"]), nn.ReLU()])
        self.dense_block = nn.Sequential(*dense_block)

        self.output_cov = model_config.get("OUTPUT_COV", False)
        n_outputs = 5 if self.output_cov else 2
        self.prob_readout = nn.Linear(model_config["LINEAR_DIM_FEEDFORWARD"], n_outputs)
        
        self.output_width = model_config["OUTPUT_WIDTH"]
        self.output_height = model_config["OUTPUT_HEIGHT"]
        self.dec_p_encoding = LearnedEncoding(
            d_model=model_config["DECODER_D_MODEL"], max_seq_len=self.output_height * self.output_width + 1,
        )


    def clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)

    def embed(self, x):
        """I'm splitting the forward pass into this 'embedding' segment
        and the final x-y coord output to make inheritance easier
        """
        # x initial shape: (batch_size, seq_len, n_mics)
        x = x.transpose(1, 2)  # (batch_size, n_mics, seq_len) for conv1d
        for layer in self.conv_layers:
            x = layer(x)  # handles the relu and skip connections
        x = x.transpose(1, 2)  # (batch_size, seq_len, d_model) for transformer
        bsz, seq, d_model = x.shape

        encoded = self.encoder(x)
        decoder_alloc = torch.zeros(size=(bsz, 1, d_model), device=x.device)
        decoder_alloc = self.dec_p_encoding(decoder_alloc)
        decoded = self.decoder(decoder_alloc, memory=encoded)
        return self.dense_block(decoded[:, 0, :])

    def forward(self, x):
        linear_out = self.embed(x)
        readout = self.prob_readout(linear_out)
        if self.output_cov:
            return build_cov_output(readout, readout.device)
        return readout
    
    def trainable_params(self):
        """Useful for freezing parts of the model"""
        return self.parameters()
