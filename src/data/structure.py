"""结构描述子：度 / 入度 / 出度 / 2 跳邻域规模（log1p）。

用于非图模型的 graph-only 与 feat+graph 设置，以及仅度数基线。
产物：data/processed/structure_features.npy（形状 n×4）
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "elliptic_bitcoin_dataset"
OUT = ROOT / "data" / "processed" / "structure_features.npy"
OUT_JSON = ROOT / "results" / "p4_structure_check.json"


def compute_structure(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    out_deg = np.bincount(src, minlength=num_nodes)
    in_deg = np.bincount(dst, minlength=num_nodes)
    degree = out_deg + in_deg

    adj = [set() for _ in range(num_nodes)]
    for a, b in zip(src, dst):
        adj[a].add(b)
        adj[b].add(a)

    two_hop = np.zeros(num_nodes, dtype=np.int64)
    for i in range(num_nodes):
        near = set(adj[i])
        for v in adj[i]:
            near |= adj[v]
        two_hop[i] = len(near)

    feats = np.stack(
        [np.log1p(degree), np.log1p(in_deg), np.log1p(out_deg), np.log1p(two_hop)],
        axis=1,
    )
    return feats.astype(np.float32)


def main() -> None:
    data = torch.load(ROOT / "data" / "processed" / "elliptic_pyg.pt", weights_only=False)
    feats = compute_structure(data.edge_index, data.x.shape[0])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, feats)
    report = {
        "shape": list(feats.shape),
        "columns": ["log1p_degree", "log1p_in_degree", "log1p_out_degree", "log1p_two_hop"],
        "mean": np.round(feats.mean(0), 4).tolist(),
        "median": np.round(np.median(feats, axis=0), 4).tolist(),
        "output": str(OUT),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
