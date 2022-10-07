from math import comb

import torch
from torch import nn
from torch.nn import functional as F

from .encodings import LearnedEncoding
from .sparse_attn import SparseTransformerEncoder, SparseTransformerEncoderLayer


class GerbilizerReducedAttentionNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_mics = config["NUM_MICROPHONES"]
        if config["COMPUTE_XCORRS"]:
            n_mics += comb(n_mics, 2)

        reduction = config["REDUCTION"]
        self.reduction = reduction
        d_model = config[f"TRANSFORMER_D_MODEL"]
        n_attn_heads = config["N_ATTENTION_HEADS"]
        n_transformer_layers = config["N_TRANSFORMER_LAYERS"]
        n_global = config["TRANSFORMER_GLOBAL_BLOCKS"]
        n_window = config["TRANSFORMER_WINDOW_BLOCKS"]
        n_random = config["TRANSFORMER_RANDOM_BLOCKS"]
        block_size = config["TRANSFORMER_BLOCK_SIZE"]

        self.data_encoding = nn.Linear(n_mics * reduction, d_model)

        encoder_layer = SparseTransformerEncoderLayer(
            d_model,
            n_attn_heads,
            block_size=block_size,
            n_global=n_global,
            n_window=n_window,
            n_random=n_random,
            dim_feedforward=2048,
            dropout=0.1,
        )

        self.transformer = SparseTransformerEncoder(encoder_layer, n_transformer_layers)

        # No dropout here since the TransformerEncoder implements its own dropout
        # Add 1 to account for the additional cls token appended in forward()
        # self.p_encoding = FixedEncoding(n_features, max_len=config['SAMPLE_LEN'] + 1)
        self.p_encoding = LearnedEncoding(
            d_model=d_model, max_seq_len=config["SAMPLE_LEN"] // reduction + 1
        )

        self.linear_dim = 1024

        self.dense = nn.Sequential(nn.Linear(d_model, self.linear_dim), nn.ReLU())
        if config["TRANSFORMER_USE_TANH"]:
            self.coord_readout = nn.Sequential(nn.Linear(self.linear_dim, 2), nn.Tanh())
        else:
            self.coord_readout = nn.Linear(self.linear_dim, 2)

    def clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)

    def embed(self, x):
        """I'm splitting the forward pass into this 'embedding' segment
        and the final x-y coord output to make inheritance easier for the
        hourglass subclass.
        """
        # X initial shape: (batch, channels, seq)
        x = x.view(x.shape[0], x.shape[1] * self.reduction, -1).transpose(-1, -2)
        reduced_x = self.data_encoding(
            x
        )  # Should have shape (batch, seq // reduction, d_model)
        cls_token = torch.zeros(
            (reduced_x.shape[0], 1, reduced_x.shape[-1]), device=reduced_x.device
        )
        transformer_input = torch.cat([cls_token, reduced_x], dim=1)
        encoded = self.p_encoding(transformer_input)
        transformer_out = self.transformer(encoded)[:, 0, :]
        return self.dense(transformer_out)

    def forward(self, x):
        linear_out = self.embed(x)
        return self.coord_readout(linear_out)

    def trainable_params(self):
        return self.parameters()


class GerbilizerAttentionHourglassNet(GerbilizerReducedAttentionNet):
    def __init__(self, config):
        super(GerbilizerAttentionHourglassNet, self).__init__(config)

        del self.coord_readout

        self.upscale = nn.Sequential(
            nn.Linear(self.linear_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 8192),
        )

        self.resize_height = 16
        self.resize_width = 16
        self.resize_channels = 32

        # Create a set of TransposeConv2d layers to upsample the reshaped vector
        self.tc_layers = nn.ModuleList()
        tc_dilations = config[f"TCONV_DILATIONS"]
        tc_strides = config[f"TCONV_STRIDES"]
        tc_kernel_sizes = config[f"TCONV_FILTER_SIZES"]
        tc_paddings = config[f"TCONV_PADDINGS"]
        tc_out_paddings = config[f"TCONV_OUTPUT_PADDINGS"]

        n_tc_channels = config[f"TCONV_NUM_CHANNELS"]
        n_tc_channels.insert(0, self.resize_channels)
        n_tc_channels[0] *= 4
        # Each entry is a tuple of the form (in_channels, out_channels)
        in_out_channels = zip(n_tc_channels[:-1], n_tc_channels[1:])

        for in_out, k_size, dilation, stride, padding, output_padding in zip(
            in_out_channels,
            tc_kernel_sizes,
            tc_dilations,
            tc_strides,
            tc_paddings,
            tc_out_paddings,
        ):
            in_channels, out_channels = in_out
            self.tc_layers.append(
                nn.Conv2d(
                    in_channels=in_channels // 4,
                    out_channels=out_channels,
                    kernel_size=k_size,
                    dilation=dilation,
                    stride=stride,
                    padding="same",
                )
            )

        self.tc_b_norms = nn.ModuleList()
        for n_features in n_tc_channels[:-1]:
            # I found that adding batchnorm helps the transpose convolutions a lot
            self.tc_b_norms.append(
                nn.BatchNorm2d(n_features // 4)
                if config["USE_BATCH_NORM"]
                else nn.Identity()
            )

    def trainable_params(self):
        # Pytorch expects a list containing a dict in which the key 'params' is assigned to a list of tensors
        return self.parameters()
        return [
            {
                "params": list(
                    chain(
                        self.upscale.parameters(),
                        self.tc_layers.parameters(),
                        self.tc_b_norms.parameters(),
                    )
                )
            }
        ]

    def forward(self, x):
        linear_out = self.upscale(super().embed(x))
        tc_output = linear_out.reshape(
            (-1, self.resize_channels, self.resize_height, self.resize_width)
        )

        for n, (tc_layer, b_norm) in enumerate(zip(self.tc_layers, self.tc_b_norms)):
            tc_output = b_norm(tc_output)
            tc_output = tc_layer(tc_output)
            tc_output = F.pixel_shuffle(tc_output, 2)
            if n < len(self.tc_layers) - 1:
                tc_output = F.relu(tc_output)
        return tc_output.squeeze()
