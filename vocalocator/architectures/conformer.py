import torch
from torch import nn
from torchaudio.models import Conformer

from ..architectures.base import VocalocatorArchitecture
from ..outputs import ModelOutputFactory
from .resnet1d import ResNet1D


def gradsafe_sum(x_list: list[torch.Tensor]) -> torch.Tensor:
    result = x_list[0]
    for x in x_list[1:]:
        result = result + x
    return result


class ResnetConformer(VocalocatorArchitecture):
    defaults = {
        "RESNET_NUM_BLOCKS": 10,
        "RESNET_CONV_CHANNELS": 64,
        "RESNET_KERNEL_SIZE": 7,
        "CONFORMER_NUM_LAYERS": 12,
        "CONFORMER_KERNEL_SIZE": 11,
        "CONFORMER_HEADS": 4,
        "CONFORMER_MLP_DIM": 512,
        "XCORR_PAIRS": None,
        "XCORR_LENGTH": 256,
        "XCORR_HIDDEN": 512,
    }

    def __init__(self, config: dict, output_factory: ModelOutputFactory):
        super().__init__(config, output_factory)

        N = config["DATA"]["NUM_MICROPHONES"]

        # Obtains model-specific parameters from the config file and fills in missing entries with defaults
        model_config = ResnetConformer.defaults.copy()
        model_config.update(config.get("MODEL_PARAMS", {}))
        config["MODEL_PARAMS"] = (
            model_config  # Save the parameters used in this run for backward compatibility
        )

        resnet_conv_channels = model_config["RESNET_CONV_CHANNELS"]
        resnet_num_blocks = model_config["RESNET_NUM_BLOCKS"]
        resnet_kernel_size = model_config["RESNET_KERNEL_SIZE"]

        conf_numheads = model_config["CONFORMER_HEADS"]
        conf_numlayers = model_config["CONFORMER_NUM_LAYERS"]
        conf_mlp_dim = model_config["CONFORMER_MLP_DIM"]
        conf_ksize = model_config["CONFORMER_KERNEL_SIZE"]

        self.resnet = ResNet1D(
            in_channels=4,
            base_filters=resnet_conv_channels // 2 ** (resnet_num_blocks // 8),
            kernel_size=resnet_kernel_size,
            stride=2,
            groups=1,
            n_block=resnet_num_blocks,
            downsample_gap=4,
            increasefilter_gap=8,
        )
        self.conformer = Conformer(
            input_dim=resnet_conv_channels,
            num_heads=conf_numheads,
            num_layers=conf_numlayers,
            ffn_dim=conf_mlp_dim,
            depthwise_conv_kernel_size=conf_ksize,
        )

        self.n_outputs: int
        self.dense = nn.Sequential(
            nn.Linear(resnet_conv_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_outputs),
        )

    def _forward(self, x):
        x = torch.einsum("btc->bct", x)  # transpose

        resnet_output = self.resnet(x)

        conformer_input = resnet_output.permute(
            0, 2, 1
        )  # (batch, channels, time) -> (batch, time, channels)
        lengths = torch.full(
            (conformer_input.size(0),), conformer_input.size(1), dtype=torch.int64
        ).to(conformer_input.device)
        conformer_output, _ = self.conformer(conformer_input, lengths)
        conformer_output = conformer_output.mean(dim=1)
        output = self.dense(conformer_output)
        return output

    def clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)
