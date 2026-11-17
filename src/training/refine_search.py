"""P5b：统一时间验证集选型与小规模消融复核。

- 所有选型只使用 train 1–29 / val 30–34 的 AP，测试集不参与。
- LR：4 个独特 C 配置；原 24 trials 中的重复仅作流程验证。
- RF：从原搜索空间无放回抽取 24 个独特配置，并纳入主搜索最佳配置。
- GCN/SAGE/GAT：主搜索 5 折结果仅用于初筛，Top-K 独特配置在时间验证集复选定型。
- GNN/FT 其他消融设置：固定主配置并补充 4–6 个独特敏感参数配置。
- 主固定配置消融回答“同容量、同超参数下的信息增量”；独立搜索结果只称“预算内最佳”。

用法：
  python src/training/refine_search.py --stage baseline --families LR,RF
  python src/training/refine_search.py --stage holdout --families GCN,SAGE,GAT
  python src/training/refine_search.py --stage sensitive --families GCN,SAGE,GAT,FT-Trans
"""

import argparse
import csv
import hashlib
import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.features import SETTINGS, get_masks
from src.training.train_loop import TrainConfig, run_experiment
from src.training.tuner import FAMILIES

MAIN_CSV = ROOT / "results" / "tuning_grid.csv"
OUT_CSV = ROOT / "results" / "refinement_grid.csv"
BEST_JSON = ROOT / "results" / "refined_best.json"
GNN_FAMILIES = ("GCN", "SAGE", "GAT")
TABULAR_FAMILIES = ("LR", "RF")
GNN_SUPPLEMENT_SETTINGS = ("local-only", "feat-only", "graph-only")
FT_SUPPLEMENT_SETTINGS = ("local-only", "feat-only", "graph-only", "degree-only")
FIELDS = [
    "stage", "family", "setting", "config_id", "params", "val_ap", "reference_ap",
    "best_epoch", "epochs_run", "seconds", "peak_mem_mb", "n_params", "status",
]


def canonical(params):
    return json.dumps(params, sort_keys=True, ensure_ascii=False)


def config_id(params):
    return hashlib.sha1(canonical(params).encode("utf-8")).hexdigest()[:12]


def read_rows(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_row(row):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def main_rows(family, require_complete=True):
    rows = [
        row for row in read_rows(MAIN_CSV)
        if row["family"] == family and row["status"] == "ok"
    ]
    expected = FAMILIES[family]["trials"]
    if require_complete and len(rows) < expected:
        raise RuntimeError(f"{family} 主搜索仅完成 {len(rows)}/{expected}，暂不能复核定型")
    return rows


def unique_main_configs(family):
    unique = {}
    for row in main_rows(family):
        key = canonical(json.loads(row["params"]))
        if key not in unique or float(row["val_ap"]) > float(unique[key]["val_ap"]):
            unique[key] = row
    return sorted(unique.values(), key=lambda row: float(row["val_ap"]), reverse=True)


def best_main_row(family):
    rows = unique_main_configs(family)
    if not rows:
        raise RuntimeError(f"{family} 没有可用的主搜索结果")
    return rows[0]


def baseline_configs(family):
    if family == "LR":
        return [{"C": value} for value in (0.01, 0.1, 1.0, 10.0)]

    full_space = [
        {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        }
        for n_estimators, max_depth, min_samples_leaf, max_features in itertools.product(
            (300, 500, 800), (None, 10, 20), (1, 2, 5), ("sqrt", 0.3)
        )
    ]
    best = json.loads(best_main_row("RF")["params"])
    remaining = [params for params in full_space if canonical(params) != canonical(best)]
    random.Random(42).shuffle(remaining)
    return [best] + remaining[:23]


def sensitive_configs(base, count=6):
    """只调整 lr / dropout / weight_decay，容量与其余训练配置保持不变。"""
    variants = [
        {},
        {"lr": base["lr"] / 3},
        {"lr": base["lr"] * 3},
        {"dropout": round(max(0.05, base["dropout"] - 0.1), 2)},
        {"dropout": round(min(0.6, base["dropout"] + 0.1), 2)},
        {"weight_decay": 1e-4 if base["weight_decay"] < 1e-4 else 0.0},
        {"lr": base["lr"] / 2, "dropout": round(min(0.6, base["dropout"] + 0.1), 2)},
        {"lr": base["lr"] * 2, "weight_decay": 1e-4 if base["weight_decay"] < 1e-4 else 0.0},
    ]
    configs, seen = [], set()
    for changes in variants:
        params = dict(base)
        params.update(changes)
        key = canonical(params)
        if key not in seen:
            configs.append(params)
            seen.add(key)
        if len(configs) == count:
            return configs
    raise RuntimeError(f"只能生成 {len(configs)} 个独特配置，无法达到 {count} 个")


def done_keys():
    return {
        (row["stage"], row["family"], row["setting"], row["config_id"])
        for row in read_rows(OUT_CSV)
        if row["status"] == "ok"
    }


def evaluate(stage, family, setting, params, reference_ap, data, structure, masks, device, dry_run):
    cid = config_id(params)
    if dry_run:
        print(f"[dry-run] {stage} {family} {setting} {cid} {canonical(params)}", flush=True)
        return

    cfg_kwargs = {} if family in TABULAR_FAMILIES else dict(params)
    if family not in TABULAR_FAMILIES:
        cfg_kwargs["epochs"] = FAMILIES[family]["epochs"]
        cfg_kwargs["patience"] = FAMILIES[family]["patience"]
    cfg = TrainConfig(seed=42, **cfg_kwargs)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    res = run_experiment(
        family, data, structure, "time", setting, cfg,
        device=device, masks=masks,
        run_tag=f"refine_{stage}_{family}_{setting}_{cid}",
        eval_test=False, save_ckpt=False,
        model_params=params if family in TABULAR_FAMILIES else None,
    )
    if "val_ap_best" in res:
        val_ap = res["val_ap_best"]
        best_epoch = res["best_epoch"]
        epochs_run = res["epochs_run"]
    else:
        val_ap = res["val_metrics"]["ap"]
        best_epoch = 0
        epochs_run = 0
    seconds = time.perf_counter() - started
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    append_row({
        "stage": stage,
        "family": family,
        "setting": setting,
        "config_id": cid,
        "params": canonical(params),
        "val_ap": round(float(val_ap), 6),
        "reference_ap": "" if reference_ap is None else round(float(reference_ap), 6),
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "seconds": round(seconds, 1),
        "peak_mem_mb": round(peak_mb, 0),
        "n_params": res.get("params") or "",
        "status": "ok",
    })
    print(
        f"[{stage}] {family} {setting} {cid} val_ap={val_ap:.6f} "
        f"({seconds:.1f}s) {canonical(params)}",
        flush=True,
    )


def write_best_summary():
    grouped = {}
    for row in read_rows(OUT_CSV):
        if row["status"] != "ok":
            continue
        key = (row["family"], row["setting"])
        if key not in grouped or float(row["val_ap"]) > float(grouped[key]["val_ap"]):
            grouped[key] = row

    # FT 主搜索本来就在固定时间验证集上进行，无需重复训练 feat+graph。
    ft_rows = [row for row in read_rows(MAIN_CSV) if row["family"] == "FT-Trans" and row["status"] == "ok"]
    if len(ft_rows) >= FAMILIES["FT-Trans"]["trials"]:
        row = best_main_row("FT-Trans")
        grouped[("FT-Trans", "feat+graph")] = {
            "family": "FT-Trans", "setting": "feat+graph", "stage": "main-time-holdout",
            "config_id": config_id(json.loads(row["params"])), "params": row["params"],
            "val_ap": row["val_ap"],
        }

    results = {}
    for (family, setting), row in sorted(grouped.items()):
        results.setdefault(family, {})[setting] = {
            "params": json.loads(row["params"]),
            "val_ap": float(row["val_ap"]),
            "source_stage": row["stage"],
            "config_id": row["config_id"],
        }
    payload = {
        "selection_protocol": {
            "train_time_steps": "1-29",
            "validation_time_steps": "30-34",
            "metric": "average_precision",
            "test_used_for_selection": False,
            "cv_role": "仅用于训练期稳定性参考与 GNN 候选初筛，不作跨模型最终排名",
            "conclusion_scope": "固定配置消融与预算内各设置最佳结果分开报告",
        },
        "results": results,
    }
    BEST_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_names(value):
    return {name.strip() for name in value.split(",") if name.strip()} if value else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("baseline", "holdout", "sensitive"))
    parser.add_argument("--families", default=None, help="逗号分隔；默认使用该阶段的全部适用模型")
    parser.add_argument("--settings", default=None, help="逗号分隔；holdout 固定为 feat+graph")
    parser.add_argument("--top-k", type=int, default=4, help="GNN 主搜索独特配置的复核数量")
    parser.add_argument("--variants", type=int, default=6, choices=(4, 5, 6))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    default_families = {
        "baseline": TABULAR_FAMILIES,
        "holdout": GNN_FAMILIES,
        "sensitive": GNN_FAMILIES + ("FT-Trans",),
    }
    families = list(parse_names(args.families) or default_families[args.stage])
    requested_settings = parse_names(args.settings)
    if requested_settings and not requested_settings.issubset(set(SETTINGS)):
        raise SystemExit(f"未知消融设置：{sorted(requested_settings)}")

    jobs = []
    if args.stage == "baseline":
        invalid = set(families) - set(TABULAR_FAMILIES)
        if invalid:
            raise SystemExit(f"baseline 阶段不支持：{sorted(invalid)}")
        for family in families:
            settings = requested_settings or set(SETTINGS)
            for setting in sorted(settings):
                for params in baseline_configs(family):
                    jobs.append((family, setting, params, None))
    elif args.stage == "holdout":
        invalid = set(families) - set(GNN_FAMILIES)
        if invalid or (requested_settings and requested_settings != {"feat+graph"}):
            raise SystemExit("holdout 仅支持 GCN/SAGE/GAT 的 feat+graph")
        for family in families:
            candidates = unique_main_configs(family)
            if len(candidates) < args.top_k:
                raise RuntimeError(f"{family} 仅有 {len(candidates)} 个独特主配置，不足 Top-{args.top_k}")
            for row in candidates[:args.top_k]:
                jobs.append((family, "feat+graph", json.loads(row["params"]), float(row["val_ap"])))
    else:
        invalid = set(families) - set(GNN_FAMILIES + ("FT-Trans",))
        if invalid:
            raise SystemExit(f"sensitive 阶段不支持：{sorted(invalid)}")
        for family in families:
            if family == "FT-Trans":
                base = json.loads(best_main_row(family)["params"])
                settings = requested_settings or set(FT_SUPPLEMENT_SETTINGS)
                invalid_settings = settings - set(FT_SUPPLEMENT_SETTINGS)
            else:
                holdout = [
                    row for row in read_rows(OUT_CSV)
                    if row["stage"] == "holdout" and row["family"] == family
                    and row["setting"] == "feat+graph" and row["status"] == "ok"
                ]
                if not holdout:
                    raise RuntimeError(f"{family} 缺少 holdout 复核结果，请先运行 --stage holdout")
                base = json.loads(max(holdout, key=lambda row: float(row["val_ap"]))["params"])
                settings = requested_settings or set(GNN_SUPPLEMENT_SETTINGS)
                invalid_settings = settings - set(GNN_SUPPLEMENT_SETTINGS)
            if invalid_settings:
                raise SystemExit(f"{family} 不应补充 feat+graph 或未知设置：{sorted(invalid_settings)}")
            for setting in sorted(settings):
                for params in sensitive_configs(base, args.variants):
                    jobs.append((family, setting, params, None))

    completed = done_keys()
    pending = [
        job for job in jobs
        if (args.stage, job[0], job[1], config_id(job[2])) not in completed
    ]
    print(f"stage={args.stage} planned={len(jobs)} completed={len(jobs) - len(pending)} pending={len(pending)}", flush=True)

    data = structure = masks = None
    device = "cpu"
    if not args.dry_run and pending:
        data = torch.load(ROOT / "data" / "processed" / "elliptic_pyg.pt", weights_only=False)
        structure = np.load(ROOT / "data" / "processed" / "structure_features.npy")
        tr, va, _ = get_masks(data, "time")
        # 第三个位置也传验证掩码，确保 sklearn 路径连测试特征转换都不会执行。
        masks = (tr, va, va)
        device = "cuda" if torch.cuda.is_available() else "cpu"

    for family, setting, params, reference_ap in pending:
        evaluate(
            args.stage, family, setting, params, reference_ap,
            data, structure, masks, device, args.dry_run,
        )
    if not args.dry_run:
        write_best_summary()
        print(f"结果：{OUT_CSV}\n预算内最佳：{BEST_JSON}", flush=True)


if __name__ == "__main__":
    main()
