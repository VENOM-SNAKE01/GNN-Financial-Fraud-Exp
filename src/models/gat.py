"""GAT：3 层 × 4 heads（每头 64，合计 256）+ 残差 + JK-Net。"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(
            [
                GATConv(
                    hidden,
                    hidden // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.jk = nn.Linear(hidden * (layers + 1), hidden)
        self.head = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.input_proj(x)
        states = [h]
        for conv, norm in zip(self.convs, self.norms):
            out = F.relu(norm(conv(h, edge_index)))
            out = F.dropout(out, p=self.dropout, training=self.training)
            h = h + out
            states.append(h)
        z = F.relu(self.jk(torch.cat(states, dim=-1)))
        z = F.dropout(z, p=self.dropout, training=self.training)
        return self.head(z).squeeze(-1)
