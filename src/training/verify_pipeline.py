"""P5 启动前链路验证（不污染正式调参 CSV）。

检查：
1. 单 trial 能完成并把结果写入独立 CSV
2. 重复启动时跳过已完成 trial
3. checkpoint 能重新加载并复现验证集 AP
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import src.training.tuner as tuner
from src.training.train_loop import TrainConfig, evaluate_checkpoint, run_experiment

OUT = ROOT / "results" / "p5_verify.json"


def main() -> None:
    verify_csv = ROOT / "results" / "verify_tuning.csv"
    verify_db = ROOT / "results" / "verify_tuning.db"
    for p in (verify_csv, verify_db):
        if p.exists():
            p.unlink()
    tuner.CSV_PATH = verify_csv
    tuner.STORAGE = f"sqlite:///{verify_db.as_posix()}"

    data = torch.load(ROOT / "data" / "processed" / "elliptic_pyg.pt", weights_only=False)
    structure = np.load(ROOT / "data" / "processed" / "structure_features.npy")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checks = {}

    tuner.tune_family(
        "GCN", data, structure, device=device, n_trials=1, epochs=3, folds=1,
    )
    rows_after_first = verify_csv.read_text(encoding="utf-8").splitlines()
    checks["trial_written"] = len(rows_after_first) == 2  # header + 1

    tuner.tune_family(
        "GCN", data, structure, device=device, n_trials=1, epochs=3, folds=1,
    )
    rows_after_second = verify_csv.read_text(encoding="utf-8").splitlines()
    checks["resume_skips_done"] = len(rows_after_second) == len(rows_after_first)

    cfg = TrainConfig(seed=42, hidden=128, layers=2, dropout=0.3, epochs=3, patience=3)
    res = run_experiment(
        "GCN", data, structure, "time", "feat+graph", cfg, device=device,
        run_tag="verify_gcn_ckpt", eval_test=False, save_ckpt=True,
    )
    reproduced = evaluate_checkpoint(
        "GCN", data, structure, "time", "feat+graph", cfg, res["checkpoint"], device=device
    )
    checks["ckpt_reload_reproduce"] = abs(reproduced - res["val_ap_best"]) < 0.02
    checks["val_only_no_test"] = res["metrics"] is None

    report = {
        "checks": checks,
        "passed": all(checks.values()),
        "trial_row": None if not checks["trial_written"] else rows_after_first[-1],
        "ckpt_val_ap": res["val_ap_best"],
        "reproduced_val_ap": round(reproduced, 6),
        "reproduce_delta": round(abs(reproduced - res["val_ap_best"]), 6),
        "note": "GCN 全图消息传递含非确定性 scatter，复算 AP 允许 <0.02 的数值差异",
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
