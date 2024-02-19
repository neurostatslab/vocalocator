from pathlib import Path

import torch
from torch import nn

from gerbilizer.architectures.base import GerbilizerArchitecture
from gerbilizer.outputs import ModelOutputFactory

from .simplenet import GerbilizerSimpleLayer


class GerbilizerNormedSimpleNetwork(GerbilizerArchitecture):
    defaults = {
        "USE_BATCH_NORM": True,
        "SHOULD_DOWNSAMPLE": [False, True] * 5,
        "CONV_FILTER_SIZES": [33] * 10,
        "CONV_NUM_CHANNELS": [16, 16, 32, 32, 64, 64, 128, 128, 256, 256],
        "CONV_DILATIONS": [1] * 10,
    }

    def __init__(self, CONFIG, output_factory: ModelOutputFactory):
        super(GerbilizerNormedSimpleNetwork, self).__init__(CONFIG, output_factory)

        N = CONFIG["DATA"]["NUM_MICROPHONES"]

        self.is_finetune = (
            "WEIGHTS_PATH" in CONFIG and Path(CONFIG["WEIGHTS_PATH"]).exists()
        )

        # Obtains model-specific parameters from the config file and fills in missing entries with defaults
        model_config = GerbilizerNormedSimpleNetwork.defaults.copy()
        model_config.update(CONFIG.get("MODEL_PARAMS", {}))
        CONFIG["MODEL_PARAMS"] = (
            model_config  # Save the parameters used in this run for backward compatibility
        )

        should_downsample = model_config["SHOULD_DOWNSAMPLE"]
        self.n_channels = model_config["CONV_NUM_CHANNELS"]
        filter_sizes = model_config["CONV_FILTER_SIZES"]
        dilations = model_config["CONV_DILATIONS"]

        min_len = min(
            len(self.n_channels),
            len(filter_sizes),
            len(should_downsample),
            len(dilations),
        )
        self.n_channels = self.n_channels[:min_len]
        filter_sizes = filter_sizes[:min_len]
        should_downsample = should_downsample[:min_len]
        dilations = dilations[:min_len]

        use_batch_norm = model_config["USE_BATCH_NORM"]

        self.n_channels.insert(0, N)
        self.init_norm = nn.BatchNorm1d(N)

        convolutions = [
            GerbilizerSimpleLayer(
                in_channels,
                out_channels,
                filter_size,
                downsample=downsample,
                dilation=dilation,
                use_bn=use_batch_norm,
            )
            for in_channels, out_channels, filter_size, downsample, dilation in zip(
                self.n_channels[:-1],
                self.n_channels[1:],
                filter_sizes,
                should_downsample,
                dilations,
            )
        ]
        self.conv_layers = torch.nn.Sequential(*convolutions)

        self.final_pooling = nn.AdaptiveAvgPool1d(1)

        if not isinstance(self.n_outputs, int):
            raise ValueError(
                "Number of parameters to output is undefined! Maybe check the model configuration and ModelOutputFactory object?"
            )
        self.coord_readout = torch.nn.Linear(self.n_channels[-1], self.n_outputs)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(
            -1, -2
        )  # (batch, seq_len, channels) -> (batch, channels, seq_len) needed by conv1d
        x = self.init_norm(x)
        h1 = self.conv_layers(x)
        h2 = torch.squeeze(self.final_pooling(h1), dim=-1)
        coords = self.coord_readout(h2)
        return coords

    def clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)

    def trainable_params(self):
        if self.is_finetune:
            # only train init_norm
            print(
                "GerbilizerNormedSimpleNetwork: Only training initial normalization layer"
            )
            return self.init_norm.parameters()

        return self.parameters()

    def train(self, mode=True):
        super().train(mode)

        if mode:
            # Freeze all batch norms except the first one
            for child in self.children():
                if isinstance(child, nn.BatchNorm1d):
                    child.eval()
                    child.weight.requires_grad = False
                    child.bias.requires_grad = False
            self.init_norm.train()
