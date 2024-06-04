import torch
from torch import nn

from vocalocator.architectures.base import VocalocatorArchitecture
from vocalocator.outputs import ModelOutputFactory


class WavenetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_channels: int,
        kernel_size: int,
        dilation: int,
        bias=True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=conv_channels * 2,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            bias=bias,
            padding="same",
        )
        self.one_by_one = nn.Conv1d(
            in_channels=conv_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor):
        conv_out = self.conv(x)
        tanh, sigmoid = conv_out.chunk(2, dim=-2)
        tanh = torch.tanh(tanh) * 0.95 + 0.05 * tanh
        sigmoid = torch.sigmoid(sigmoid)
        activation = tanh * sigmoid
        one_by_one_output = self.one_by_one(activation)
        return one_by_one_output + x, one_by_one_output


def gradsafe_sum(x_list: list[torch.Tensor]) -> torch.Tensor:
    result = x_list[0]
    for x in x_list[1:]:
        result = result + x
    return result


class Wavenet(VocalocatorArchitecture):
    defaults = {
        "NUM_BLOCKS": 10,
        "CONV_CHANNELS": 64,
        "KERNEL_SIZE": 7,
        "DILATION": 3,
        "OUTPUT_COV": True,
    }

    def __init__(self, config: dict, output_factory: ModelOutputFactory):
        super().__init__(config, output_factory)

        N = config["DATA"]["NUM_MICROPHONES"]
        if "CHANNELS_TO_USE" in config["DATA"]:
            N = len(config["DATA"]["CHANNELS_TO_USE"])

        # Obtains model-specific parameters from the config file and fills in missing entries with defaults
        model_config = Wavenet.defaults.copy()
        model_config.update(config.get("MODEL_PARAMS", {}))
        config["MODEL_PARAMS"] = (
            model_config  # Save the parameters used in this run for backward compatibility
        )

        self.blocks = nn.ModuleList(
            [
                WavenetBlock(
                    in_channels=model_config["CONV_CHANNELS"],
                    conv_channels=model_config["CONV_CHANNELS"],
                    kernel_size=model_config["KERNEL_SIZE"],
                    dilation=i % 4 + 1,
                )
                for i in range(model_config["NUM_BLOCKS"])
            ]
        )
        self.initial_conv = nn.Conv1d(
            in_channels=N,
            out_channels=model_config["CONV_CHANNELS"],
            kernel_size=model_config["KERNEL_SIZE"],
            stride=1,
            dilation=1,
            padding="same",
        )

        self.n_outputs: int
        self.dense = nn.Sequential(
            nn.Linear(model_config["CONV_CHANNELS"], 512),
            nn.ReLU(),
            nn.Linear(512, self.n_outputs),
        )

    def _forward(self, x):
        x = torch.einsum("btc->bct", x)
        output = self.initial_conv(x)
        one_by_ones = []
        for block in self.blocks:
            output, obo = block(output)
            one_by_ones.append(obo)
        # Mean over time dim
        obo_sum = gradsafe_sum(one_by_ones).mean(dim=-1)
        output = self.dense(obo_sum)
        return output

    def clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)
