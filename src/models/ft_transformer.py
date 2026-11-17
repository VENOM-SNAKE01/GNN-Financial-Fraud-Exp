"""FT-Transformer：逐特征 tokenizer + 6 层 TransformerEncoder + [CLS] 分类。

每个特征一个标量，因此 per-feature linear 退化为 x_j * w_j + b_j（可学习的 per-feature embedding）。
"""

import torch
from torch import nn


class FTTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_token: int = 192,
        layers: int = 6,
        heads: int = 8,
        dropout: float = 0.1,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, d_token))
        self.bias = nn.Parameter(torch.zeros(in_dim, d_token))
        nn.init.xavier_uniform_(self.weight)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=heads,
            dim_feedforward=d_token * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x):
        # x: (B, n_features)
        tok = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls.expand(x.size(0), -1, -1)
        z = torch.cat([cls, tok], dim=1)
        z = self.encoder(z)
        z = self.norm(z[:, 0])
        return self.head(z).squeeze(-1)
