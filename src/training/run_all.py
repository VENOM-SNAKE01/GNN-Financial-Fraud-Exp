"""P5 总入口：按族调参，支持断点续跑。

用法：
  python src/training/run_all.py --families GCN --trials 1 --epochs 3 --folds 1
  python src/training/run_all.py            # 按各模型族默认预算跑
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.training.tuner import CSV_PATH, FAMILIES, best_params, tune_family

# 主搜索中的 CV 结果只用于初筛；最终选型见 refine_search.py / refined_best.json。
OUT = ROOT / "results" / "tuning_best_cv.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="LR,RF,GCN,SAGE,GAT,FT-Trans")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--setting", default="feat+graph")
    args = ap.parse_args()

    data = torch.load(ROOT / "data" / "processed" / "elliptic_pyg.pt", weights_only=False)
    structure = np.load(ROOT / "data" / "processed" / "structure_features.npy")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    for fam in families:
        if fam not in FAMILIES:
            raise SystemExit(f"unknown family {fam}")
        print(f"=== {fam} ===", flush=True)
        tune_family(
            fam, data, structure, device=device, n_trials=args.trials,
            epochs=args.epochs, folds=args.folds, setting=args.setting,
        )

    summary = {fam: best_params(fam, CSV_PATH) for fam in families}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
