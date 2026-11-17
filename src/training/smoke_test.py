"""P4 烟雾测试：feat-only 跑通 3 epoch，loss 下降；并验证 feat+graph 与结构描述子路径。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.training.train_loop import LOG_DIR, TrainConfig, run_experiment

OUT = ROOT / "results" / "p4_smoke.json"


def read_log(tag: str):
    path = LOG_DIR / f"{tag}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {
        "epochs": len(rows),
        "first_loss": rows[0]["train_loss"],
        "last_loss": rows[-1]["train_loss"],
        "first_val_ap": rows[0]["val_ap"],
        "last_val_ap": rows[-1]["val_ap"],
    }


def main() -> None:
    data = torch.load(ROOT / "data" / "processed" / "elliptic_pyg.pt", weights_only=False)
    structure = np.load(ROOT / "data" / "processed" / "structure_features.npy")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    plan = [
        ("GCN", "feat-only", dict(hidden=256, layers=4, dropout=0.4)),
        ("GCN", "feat+graph", dict(hidden=256, layers=4, dropout=0.4)),
        ("SAGE", "feat-only", dict(hidden=256, layers=4, dropout=0.4)),
        ("SAGE", "feat+graph", dict(hidden=256, layers=4, dropout=0.4)),
        ("GAT", "feat-only", dict(hidden=256, layers=3, dropout=0.5)),
        ("GAT", "feat+graph", dict(hidden=256, layers=3, dropout=0.5)),
        ("LR", "feat-only", {}),
        ("LR", "graph-only", {}),
        ("RF", "feat-only", {}),
        ("FT-Trans", "feat-only", dict(hidden=192, layers=6, dropout=0.1, batch_size=256)),
        ("FT-Trans", "feat+graph", dict(hidden=192, layers=6, dropout=0.1, batch_size=256)),
    ]
    for model, setting, overrides in plan:
        cfg = TrainConfig(epochs=3, patience=3, seed=42, **overrides)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        res = run_experiment(model, data, structure, "time", setting, cfg, device=device)
        tag = f"{model}_{setting}_time_s42"
        log_info = read_log(tag) if (LOG_DIR / f"{tag}.jsonl").exists() else {}
        info = {
            "model": model,
            "setting": setting,
            "test_ap": res["metrics"]["ap"],
            "test_auc": res["metrics"]["auc"],
            "test_f1": res["metrics"]["f1"],
            "params": res.get("params"),
            "gpu_hours": res["gpu_hours"],
            **log_info,
        }
        if device == "cuda":
            info["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        results.append(info)
        print(json.dumps(info, ensure_ascii=False), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
