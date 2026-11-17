"""P5 主搜索（测试集不参与）。

- GNN / LR / RF：训练期 1–29 内 5 折分层交叉验证，仅作稳定性参考与候选初筛
- FT-Trans：固定单折（train 1–29 / val 30–34），12 trials，前 6 宽搜 + 后 6 集中搜索
- 最终跨模型选型统一由 refine_search.py 按 30–34 时间验证集 AP 复核
- LR 的 24 次原始运行仅含 4 个独特配置，有效搜索预算记为 4，重复不计入预算
- 每个 trial 完成立即 append 到 results/tuning_grid.csv；支持断点续跑
"""

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import optuna
import torch
from sklearn.model_selection import StratifiedKFold

from src.data.features import get_masks
from src.training.train_loop import TrainConfig, run_experiment

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = Path(os.environ.get("EXP03_TUNING_CSV", ROOT / "results" / "tuning_grid.csv"))
STORAGE = os.environ.get(
    "EXP03_TUNING_DB", f"sqlite:///{(ROOT / 'results' / 'tuning.db').as_posix()}"
)

FIELDS = [
    "family", "trial", "params", "val_ap", "fold_aps", "best_epoch",
    "epochs_run", "seconds", "peak_mem_mb", "n_params", "status",
]

FAMILIES = {
    "LR": {"trials": 4, "folds": 5, "epochs": None, "patience": None},
    "RF": {"trials": 24, "folds": 5, "epochs": None, "patience": None},
    "GCN": {"trials": 24, "folds": 5, "epochs": 500, "patience": 30},
    "SAGE": {"trials": 24, "folds": 5, "epochs": 500, "patience": 30},
    "GAT": {"trials": 24, "folds": 5, "epochs": 500, "patience": 30},
    "FT-Trans": {"trials": 12, "folds": 1, "epochs": 60, "patience": 15},
}


def build_folds(y, tr, va, n_splits, seed=42):
    if n_splits <= 1:
        return [(tr.copy(), va.copy())]
    idx = np.flatnonzero(tr)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fit, fva in skf.split(idx, y[idx]):
        m_tr = np.zeros(len(y), dtype=bool)
        m_va = np.zeros(len(y), dtype=bool)
        m_tr[idx[fit]] = True
        m_va[idx[fva]] = True
        folds.append((m_tr, m_va))
    return folds


def _suggest_shared(trial):
    return {
        "lr": trial.suggest_categorical("lr", [1e-4, 3e-4, 1e-3, 3e-3]),
        "hidden": trial.suggest_categorical("hidden", [128, 256, 512]),
        "layers": trial.suggest_categorical("layers", [2, 4, 6]),
        "dropout": trial.suggest_categorical("dropout", [0.2, 0.3, 0.5]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4]),
    }


def _suggest_ft(trial, study):
    """前 6 次宽搜，后 6 次围绕当前最优区域收窄。"""
    stage2 = trial.number >= 6 and len(study.trials) >= 6
    if not stage2:
        d_token = trial.suggest_categorical("d_token", [128, 192, 256])
        n_layers = trial.suggest_categorical("n_layers", [2, 4, 6])
        dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3])
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    else:
        best = study.best_trial.params
        d_token = trial.suggest_categorical(
            "d_token", sorted({128, int(best.get("d_token", 192)), 256})
        )
        n_layers = trial.suggest_categorical(
            "n_layers", sorted({2, int(best.get("n_layers", 4)), 6})
        )
        base_drop = float(best.get("dropout", 0.2))
        drop_opts = sorted({max(0.05, round(base_drop - 0.1, 2)), base_drop,
                            min(0.5, round(base_drop + 0.1, 2))})
        dropout = trial.suggest_categorical("dropout", drop_opts)
        base_lr = float(best.get("lr", 1e-3))
        lr = trial.suggest_float("lr", max(1e-4, base_lr / 3), min(3e-3, base_lr * 3), log=True)
    n_heads = trial.suggest_categorical("n_heads", [4, 8])
    return {
        "lr": lr,
        "hidden": d_token,
        "layers": n_layers,
        "dropout": dropout,
        "n_heads": n_heads,
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4]),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512]),
        "pos_weight_scale": trial.suggest_categorical("pos_weight_scale", [0.5, 1.0, 2.0]),
    }


def _suggest_lr(trial):
    return {"C": trial.suggest_categorical("C", [0.01, 0.1, 1.0, 10.0])}


def _suggest_rf(trial):
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [300, 500, 800]),
        "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20]),
        "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 5]),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.3]),
    }


def _append_row(row: dict) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def _read_done(family: str) -> int:
    if not CSV_PATH.exists():
        return 0
    with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["family"] == family and r["status"] == "ok"]
    return len(rows)


def tune_family(
    family, data, structure, device="cuda", n_trials=None, epochs=None, folds=None,
    setting="feat+graph", smoke=False,
):
    spec = FAMILIES[family]
    n_trials = n_trials or spec["trials"]
    n_folds = folds or spec["folds"]
    max_epochs = epochs or spec["epochs"]
    patience = spec["patience"] or 30

    y = data.y.numpy()
    tr, va, _ = get_masks(data, "time")
    fold_list = build_folds(y, tr, va, n_folds)
    done = _read_done(family)
    if done >= n_trials:
        print(f"[{family}] already completed {done}/{n_trials}, skip", flush=True)
        return done

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_name = f"exp03_{family}"
    study = optuna.create_study(
        study_name=study_name,
        storage=STORAGE,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    # 已记录在 CSV 的 trial 视为完成：把 study 中对应编号也补齐
    print(f"[{family}] csv_done={done}, study_trials={len(study.trials)}", flush=True)

    def objective(trial):
        if family == "FT-Trans":
            params = _suggest_ft(trial, study)
        elif family == "LR":
            params = _suggest_lr(trial)
        elif family == "RF":
            params = _suggest_rf(trial)
        else:
            params = _suggest_shared(trial)

        cfg_kwargs = dict(params)
        model_params = {}
        if family in ("LR", "RF"):
            model_params = params
            cfg_kwargs = {}
        cfg_kwargs.setdefault("epochs", max_epochs or 500)
        cfg_kwargs.setdefault("patience", patience)
        fold_aps, best_epochs, epochs_run = [], [], []
        t0 = time.perf_counter()
        peak_mb = 0.0

        for fi, (ftr, fva) in enumerate(fold_list):
            cfg = TrainConfig(seed=42 + fi, **cfg_kwargs)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            res = run_experiment(
                family, data, structure, "time", setting, cfg, device=device,
                masks=(ftr, fva, fva), run_tag=f"tune_{family}_t{trial.number}_f{fi}",
                eval_test=False, save_ckpt=False, model_params=model_params,
            )
            if "val_ap_best" in res:
                fold_aps.append(res["val_ap_best"])
                best_epochs.append(res["best_epoch"])
                epochs_run.append(res["epochs_run"])
            else:  # sklearn 基线没有 epoch 概念
                fold_aps.append(res["val_metrics"]["ap"])
                best_epochs.append(0)
                epochs_run.append(0)
            if torch.cuda.is_available():
                peak_mb = max(peak_mb, torch.cuda.max_memory_allocated() / 2**20)

        val_ap = float(np.mean(fold_aps))
        seconds = time.perf_counter() - t0
        _append_row({
            "family": family,
            "trial": trial.number,
            "params": json.dumps(params, ensure_ascii=False),
            "val_ap": round(val_ap, 6),
            "fold_aps": json.dumps([round(a, 6) for a in fold_aps]),
            "best_epoch": int(np.mean(best_epochs)),
            "epochs_run": int(np.sum(epochs_run)),
            "seconds": round(seconds, 1),
            "peak_mem_mb": round(peak_mb, 0),
            "n_params": res.get("params") or "",
            "status": "ok",
        })
        print(f"[{family}] trial {trial.number} val_ap={val_ap:.4f} "
              f"({seconds:.0f}s, ep={epochs_run}) {params}", flush=True)
        return val_ap

    remaining = n_trials - done
    study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    return n_trials


def best_params(family, path=CSV_PATH):
    import csv as _csv

    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            if r["family"] == family and r["status"] == "ok":
                rows.append(r)
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r["val_ap"]))
    return {
        "trial": int(best["trial"]),
        "val_ap": float(best["val_ap"]),
        "params": json.loads(best["params"]),
    }
