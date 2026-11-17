"""P3：建图与两套划分。

- 节点 = 交易，特征 = 166 维（含时间步这一维），y: illicit=1 / licit=0 / unknown=-1
- 时间划分（主）：train 1–29 / val 30–34 / test 35–49，互不共享边
- 随机分层划分（辅，泄漏对照）：70/15/15，seed 42
- 产物：data/processed/elliptic_pyg.pt、results/p3_build_check.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "elliptic_bitcoin_dataset"
OUT_DATA = ROOT / "data" / "processed" / "elliptic_pyg.pt"
OUT_JSON = ROOT / "results" / "p3_build_check.json"

SEED = 42
TRAIN_STEPS = (1, 29)
VAL_STEPS = (30, 34)
TEST_STEPS = (35, 49)


def main() -> None:
    # txId 必须用 int64 读：float32 无法精确表示 2 亿级别的交易号
    dtypes = {0: np.int64, 1: np.int64}
    dtypes.update({i: np.float32 for i in range(2, 167)})
    feats = pd.read_csv(RAW / "elliptic_txs_features.csv", header=None, dtype=dtypes)

    tx_ids = feats[0].to_numpy(dtype=np.int64)
    x = feats.iloc[:, 1:].to_numpy(dtype=np.float32)
    time_step = feats[1].to_numpy(dtype=np.int64)
    classes = pd.read_csv(RAW / "elliptic_txs_classes.csv", dtype={"class": str})
    edges = pd.read_csv(RAW / "elliptic_txs_edgelist.csv")

    n = len(tx_ids)
    idx = pd.Series(np.arange(n), index=tx_ids)

    y = np.full(n, -1, dtype=np.int64)
    cls = classes.set_index("txId")["class"]
    y[idx[cls.index].to_numpy()] = cls.map({"1": 1, "2": 0}).fillna(-1).to_numpy(dtype=np.int64)

    edge_index = torch.from_numpy(
        np.vstack([idx[edges["txId1"]].to_numpy(), idx[edges["txId2"]].to_numpy()])
    ).long()
    m = edge_index.shape[1]

    # ---- 时间划分 ----
    labeled = y != -1
    in_train = labeled & (time_step >= TRAIN_STEPS[0]) & (time_step <= TRAIN_STEPS[1])
    in_val = labeled & (time_step >= VAL_STEPS[0]) & (time_step <= VAL_STEPS[1])
    in_test = labeled & (time_step >= TEST_STEPS[0]) & (time_step <= TEST_STEPS[1])

    # 时间步 -> 归属，用于"零边共享"核验
    period = np.full(n, -1, dtype=np.int64)
    period[(time_step >= TRAIN_STEPS[0]) & (time_step <= TRAIN_STEPS[1])] = 0
    period[(time_step >= VAL_STEPS[0]) & (time_step <= VAL_STEPS[1])] = 1
    period[(time_step >= TEST_STEPS[0]) & (time_step <= TEST_STEPS[1])] = 2
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    cross_edges = int(((period[src] != period[dst]) & (period[src] >= 0) & (period[dst] >= 0)).sum())
    edge_in_period = {str(k): int((period[src] == k).sum()) for k in (0, 1, 2)}

    # ---- 随机分层划分（仅已标注节点）----
    lab_idx = np.flatnonzero(labeled)
    tr_idx, rest_idx = train_test_split(
        lab_idx, test_size=0.3, random_state=SEED, stratify=y[lab_idx]
    )
    va_idx, te_idx = train_test_split(
        rest_idx, test_size=0.5, random_state=SEED, stratify=y[rest_idx]
    )
    rand_train = np.zeros(n, dtype=bool)
    rand_val = np.zeros(n, dtype=bool)
    rand_test = np.zeros(n, dtype=bool)
    rand_train[tr_idx], rand_val[va_idx], rand_test[te_idx] = True, True, True

    data = Data(
        x=torch.from_numpy(x.astype(np.float32)),
        edge_index=edge_index,
        y=torch.from_numpy(y),
        time_step=torch.from_numpy(time_step),
        node_id=torch.from_numpy(tx_ids),
        train_mask_time=torch.from_numpy(in_train),
        val_mask_time=torch.from_numpy(in_val),
        test_mask_time=torch.from_numpy(in_test),
        train_mask_random=torch.from_numpy(rand_train),
        val_mask_random=torch.from_numpy(rand_val),
        test_mask_random=torch.from_numpy(rand_test),
    )

    # ---- 验收断言 ----
    assert data.x.shape == (203_769, 166), data.x.shape
    assert data.edge_index.shape == (2, 234_355), data.edge_index.shape
    assert set(np.unique(y).tolist()) <= {-1, 0, 1}
    for name in ("train", "val", "test"):
        mt = getattr(data, f"{name}_mask_time").numpy()
        mr = getattr(data, f"{name}_mask_random").numpy()
        assert not (mt & (y == -1)).any(), f"{name}_mask_time 含未标注节点"
        assert not (mr & (y == -1)).any(), f"{name}_mask_random 含未标注节点"
    assert not (in_train & in_val).any() and not (in_val & in_test).any()
    assert not (in_train & in_test).any()
    assert (in_train | in_val | in_test).sum() == int(labeled.sum()) == 46_564
    assert (rand_train | rand_val | rand_test).sum() == 46_564
    assert cross_edges == 0, f"时间划分存在 {cross_edges} 条跨期边"
    assert np.isfinite(x).all(), "特征含 NaN/Inf"

    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, OUT_DATA)

    def split_stats(mask: np.ndarray) -> dict:
        yy = y[mask]
        return {
            "nodes": int(mask.sum()),
            "illicit": int((yy == 1).sum()),
            "licit": int((yy == 0).sum()),
            "pos_rate": round(float((yy == 1).mean()), 6),
        }

    report = {
        "data_version": "Elliptic++ 交易子集（手动下载 archive.zip）",
        "nodes": int(n),
        "edges": int(m),
        "feature_dim": int(x.shape[1]),
        "columns_note": "文件 167 列 = txId + 166 维特征（其中第 1 维为时间步 1–49）",
        "labels": {
            "illicit": int((y == 1).sum()),
            "licit": int((y == 0).sum()),
            "unknown": int((y == -1).sum()),
        },
        "time_split": {
            "train_steps": list(TRAIN_STEPS),
            "val_steps": list(VAL_STEPS),
            "test_steps": list(TEST_STEPS),
            "train": split_stats(in_train),
            "val": split_stats(in_val),
            "test": split_stats(in_test),
            "cross_period_edges": cross_edges,
            "edges_by_period": edge_in_period,
        },
        "random_split": {
            "ratio": "70/15/15",
            "seed": SEED,
            "train": split_stats(rand_train),
            "val": split_stats(rand_val),
            "test": split_stats(rand_test),
            "cross_split_edges": int(
                ((rand_train[src] & ~rand_train[dst])
                 | (rand_val[src] & ~rand_val[dst])
                 | (rand_test[src] & ~rand_test[dst])).sum()
            ),
        },
        "assertions_passed": True,
        "output": str(OUT_DATA),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
