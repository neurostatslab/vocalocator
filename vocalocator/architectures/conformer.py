import torch
from torch import nn
from torchaudio.models import Conformer

from ..architectures.base import VocalocatorArchitecture
from ..architectures.lora import LORA_MHA
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
            in_channels=N,
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

    def _make_finetuneable(self):
        """Convert all conformer layers to finetunable layers and disable gradient flow through conv layers"""
        rank = self.config["FINETUNING"]["LORA_RANK"]
        for p in self.parameters():
            p.requires_grad_(False)
        for layer in self.conformer.conformer_layers:
            layer.self_attn = LORA_MHA(layer.self_attn, rank)
            layer.self_attn.requires_grad_(True)

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


if __name__ == "__main__":
    from itertools import product

    import numpy as np
    from tqdm import tqdm

    from ..training.configs import DEFAULT_CONFIG
    from ..training.models import make_output_factory

    sample_config = DEFAULT_CONFIG.copy()
    factory = make_output_factory(sample_config)
    sample_data = torch.rand(1, 1024, 4)

    # Test various choices of model params to see what works and what doesn't
    ranges = {
        "RESNET_NUM_BLOCKS": [2, 31],
        "RESNET_CONV_CHANNELS": (32, 64, 128, 256),
        # "RESNET_KERNEL_SIZE": list(map(lambda x: 2 * x + 1, range(1, 20))),
        "RESNET_KERNEL_SIZE": [3, 41],
        # "CONFORMER_NUM_LAYERS": list(range(1, 20)),
        "CONFORMER_NUM_LAYERS": [1, 30],
        # "CONFORMER_KERNEL_SIZE": list(map(lambda x: 2 * x + 1, range(1, 20))),
        "CONFORMER_KERNEL_SIZE": (3, 31),
        "CONFORMER_HEADS": (1, 2, 4, 8, 16),
        "CONFORMER_MLP_DIM": (128, 256, 512, 1024),
    }
    keys = list(ranges.keys())  # force an ordering
    ranges = [ranges[k] for k in keys]
    prod = product(*ranges)
    num_tests = np.prod(list(map(len, ranges)))

    for configuration in tqdm(prod, total=num_tests):
        model_params = dict(zip(keys, configuration))
        cfg = sample_config.copy()
        cfg["MODEL_PARAMS"] = model_params
        # try:
        model = ResnetConformer(cfg, factory)
        model._make_finetuneable()
        model(sample_data)
        # except Exception as e:
        # print(f"{type(e)} for the following configuration:")
        # for key, value in model_params.items():
        #     print(f"\t{key}: {value}")
        # print()
        # print()
