import torch
from torch import nn
from torch.nn import functional as F

from .util import build_cov_output

from .encodings import LearnedEncoding


from itertools import chain

import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal

from .sparse_transformers import SparseTransformerEncoder, SparseTransformerEncoderLayer


class AttentionNet(nn.Module):
    default_config = {
        "CROP_SIZE": 4096,
        "BLOCK_SIZE": 16,
        "D_MODEL": 256,
        "NUM_HEADS": 8,
        "OUTPUT_COV": True,
    }

    def __init__(self, config: dict):
        """Constructs a regression model over the microphone traces
        of Mongolian gerbil vocalizations.
        Prameters:
            - crop_size: max length of the input sequence
            - block_size: number of audio samples per input token
            - d_model: inner dimension size of transformer layers
            - num_heads: Number of attention heads to use
        """
        super(AttentionNet, self).__init__()

        params = AttentionNet.default_config.copy()
        params.update(config["MODEL_PARAMS"])

        self.n_mics = config["DATA"]["NUM_MICROPHONES"]

        self.crop_size = config["DATA"]["CROP_LENGTH"]
        self.block_size = params["BLOCK_SIZE"]
        self.max_seq = self.crop_size // self.block_size + 1
        self.d_model = params["D_MODEL"]
        self.num_heads = params["NUM_HEADS"]
        self.output_cov = params["OUTPUT_COV"]

        self.cls_token = nn.Parameter(torch.zeros(1, self.d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.in_encoding = LearnedEncoding(
            d_model=self.d_model, max_seq_len=self.max_seq
        )
        self.out_encoding = LearnedEncoding(
            d_model=self.d_model, max_seq_len=self.max_seq
        )

        self.data_encoding = nn.Linear(self.block_size * self.n_mics, self.d_model)
        self.encoder = SparseTransformerEncoder(
            SparseTransformerEncoderLayer(
                self.d_model,
                self.num_heads,
                block_size=8,
                n_global=2,
                n_window=11,
                n_random=3,
                dim_feedforward=2048,
                dropout=0.1,
                checkpoint=False,
                batch_first=True,
            ),
            5,
        )

        n_output = 5 if self.output_cov else 2
        self.dense = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, n_output),
        )

    def _clip_gradients(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)

    def encode(self, x):
        batched = x.dim() == 3
        # Initial shape (batch, samples, mics)
        if batched:
            x = x.reshape(x.shape[0], -1, self.block_size * self.n_mics)
        else:
            x = x.reshape(-1, self.block_size * self.n_mics)
        # shape: (batch, blocks, block_size * mics)
        embed_out = self.data_encoding(x)  # Outputs (batch, blocks, d_model)
        cls_token = self.cls_token
        if batched:
            cls_token = cls_token.unsqueeze(0).expand(x.shape[0], -1, -1)
        transformer_in = torch.cat([cls_token, embed_out], dim=-2)
        transformer_in = self.in_encoding(
            transformer_in.unsqueeze(0) if not batched else transformer_in
        )
        transformer_out = self.encoder(transformer_in)[:, 0, :]
        return transformer_out

    def forward(self, x):
        encoded = self.dense(self.encode(x))
        if self.output_cov:
            return build_cov_output(encoded, device=x.device)
        else:
            return encoded
