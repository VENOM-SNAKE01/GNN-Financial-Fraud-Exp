"""按监督设置装配输入特征，并只用训练节点统计量做标准化。"""

import numpy as np

LOCAL_DIM = 94
SETTINGS = ("local-only", "feat-only", "feat+graph", "graph-only", "degree-only")
SETTING_LABELS = {
    "local-only": "仅 94 维 local 特征",
    "feat-only": "166 维特征、不使用边",
    "feat+graph": "166 维特征 + 图结构",
    "graph-only": "仅图结构",
    "degree-only": "仅无向度数（基线）",
}


def get_masks(data, split: str):
    return (
        data[f"train_mask_{split}"].numpy(),
        data[f"val_mask_{split}"].numpy(),
        data[f"test_mask_{split}"].numpy(),
    )


def build_feature_matrix(x: np.ndarray, structure: np.ndarray, setting: str, kind: str):
    """返回 (特征矩阵, 是否需要标准化)。kind ∈ {gnn, tabular}。"""
    if kind == "gnn":
        if setting == "graph-only":
            return np.ones_like(x, dtype=np.float32), False
        if setting == "local-only":
            return x[:, :LOCAL_DIM], True
        return x, True  # feat-only 用自环断边，feat+graph 用全图
    if setting == "local-only":
        return x[:, :LOCAL_DIM], True
    if setting == "feat-only":
        return x, True
    if setting == "feat+graph":
        return np.hstack([x, structure]).astype(np.float32), True
    if setting == "graph-only":
        return structure, True
    if setting == "degree-only":
        return structure[:, :1], True
    raise ValueError(setting)


def standardize(X: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    mu = X[train_mask].mean(axis=0)
    sd = X[train_mask].std(axis=0)
    sd[sd < 1e-6] = 1.0
    return ((X - mu) / sd).astype(np.float32)
