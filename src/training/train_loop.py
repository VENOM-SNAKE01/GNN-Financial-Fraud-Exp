"""统一训练循环：GNN 全图训练 / FT-Transformer 节点 mini-batch。

- 每个 epoch 追加一行 JSONL，崩溃可续
- 早停监控验证集 AP，patience 可配
- 调参时 eval_test=False，测试集完全不参与选型
- 验证集上选 F1 阈值，测试集只在最终评估时使用
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.utils import to_undirected

from src.data.features import build_feature_matrix, get_masks, standardize
from src.evaluation.metrics import best_f1_threshold, compute_metrics
from src.models.ft_transformer import FTTransformer
from src.models.gat import GAT
from src.models.gcn import GCN
from src.models.sage import GraphSAGE

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "results" / "logs"
CKPT_DIR = ROOT / "results" / "checkpoints"

TABULAR = ("LR", "RF")


@dataclass
class TrainConfig:
    lr: float = 1e-3
    hidden: int = 256
    layers: int = 4
    dropout: float = 0.4
    weight_decay: float = 1e-5
    epochs: int = 500
    patience: int = 30
    batch_size: int = 1024
    clip: float = 1.0
    seed: int = 42
    warmup_frac: float = 0.05
    n_heads: int = 8
    pos_weight_scale: float = 1.0


def build_model(model_name: str, in_dim: int, cfg: TrainConfig):
    if model_name == "GCN":
        return GCN(in_dim, cfg.hidden, cfg.layers, cfg.dropout)
    if model_name == "SAGE":
        return GraphSAGE(in_dim, cfg.hidden, cfg.layers, cfg.dropout)
    if model_name == "GAT":
        return GAT(in_dim, cfg.hidden, cfg.layers, heads=4, dropout=cfg.dropout)
    if model_name == "FT-Trans":
        return FTTransformer(
            in_dim,
            d_token=cfg.hidden,
            layers=cfg.layers,
            heads=cfg.n_heads,
            dropout=cfg.dropout,
        )
    raise ValueError(model_name)


def _make_scheduler(optimizer, cfg: TrainConfig, total_epochs: int):
    warmup = max(1, int(total_epochs * cfg.warmup_frac))

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total_epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _pos_weight(y_train: np.ndarray, scale: float = 1.0) -> float:
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    return neg / max(pos, 1.0) * scale


def _finalize(y, va, te, best_state, model, score_fn, tag, eval_test=True, save_ckpt=True):
    """载入最优权重；验证集选阈值；测试集仅在 eval_test 时评估。"""
    model.load_state_dict(best_state)
    model.eval()
    va_idx = np.flatnonzero(va)
    va_scores = score_fn(va_idx)
    threshold = best_f1_threshold(y[va_idx], va_scores)
    val_metrics = compute_metrics(y[va_idx], va_scores, threshold)
    metrics = None
    if eval_test:
        te_idx = np.flatnonzero(te)
        metrics = compute_metrics(y[te_idx], score_fn(te_idx), threshold)
    ckpt = None
    if save_ckpt:
        path = CKPT_DIR / f"{tag}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        ckpt = str(path)
    return metrics, val_metrics, threshold, ckpt


def _train_gnn(
    model_name, X, y, masks, edge_index, cfg, device, tag, split, setting,
    eval_test=True, save_ckpt=True,
):
    torch.manual_seed(cfg.seed)
    n = len(y)
    tr, va, te = masks
    x_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y.astype(np.float32)).to(device)
    tr_t = torch.from_numpy(tr).to(device)
    va_t = torch.from_numpy(va).to(device)

    if setting in ("feat-only", "local-only"):
        ei = torch.arange(n, device=device).repeat(2, 1)
    else:
        ei = to_undirected(edge_index, num_nodes=n).to(device)

    model = build_model(model_name, X.shape[1], cfg).to(device)
    pos_weight = torch.tensor([_pos_weight(y[tr], cfg.pos_weight_scale)], device=device)
    loss_fn = F.binary_cross_entropy_with_logits
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = _make_scheduler(opt, cfg, cfg.epochs)

    log_path = LOG_DIR / f"{tag}.jsonl"
    best_ap, best_state, best_epoch, bad = -1.0, None, 0, 0
    t0 = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x_t, ei)
        loss = loss_fn(logits[tr_t], y_t[tr_t], pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_t, ei)[va_t]
        val_scores = torch.sigmoid(val_logits).cpu().numpy()
        val_ap = float(average_precision_score(y[va], val_scores))
        _append_jsonl(log_path, {
            "epoch": epoch,
            "train_loss": round(float(loss.item()), 6),
            "val_ap": round(val_ap, 6),
            "lr": opt.param_groups[0]["lr"],
            "elapsed_s": round(time.perf_counter() - t0, 2),
        })

        if val_ap > best_ap + 1e-6:
            best_ap, best_epoch, bad = val_ap, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    def score_fn(idx):
        with torch.no_grad():
            logits = model(x_t, ei)[torch.from_numpy(idx).to(device)]
        return torch.sigmoid(logits).detach().cpu().numpy()

    metrics, val_metrics, threshold, ckpt = _finalize(
        y, va, te, best_state, model, score_fn, tag, eval_test, save_ckpt
    )
    return {
        "model": model_name,
        "setting": setting,
        "split": split,
        "seed": cfg.seed,
        "metrics": metrics,
        "val_metrics": val_metrics,
        "val_ap_best": round(best_ap, 6),
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "params": int(sum(p.numel() for p in model.parameters())),
        "gpu_hours": round((time.perf_counter() - t0) / 3600, 6),
        "checkpoint": ckpt,
    }


def _score_ft(model, x_t, idx, batch_size=256):
    outs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            batch = torch.from_numpy(idx[start : start + batch_size]).to(x_t.device)
            outs.append(torch.sigmoid(model(x_t[batch])).cpu().numpy())
    return np.concatenate(outs)


def _train_ft(
    model_name, X, y, masks, cfg, device, tag, split, setting,
    eval_test=True, save_ckpt=True,
):
    torch.manual_seed(cfg.seed)
    tr, va, te = masks
    x_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y.astype(np.float32)).to(device)

    model = build_model(model_name, X.shape[1], cfg).to(device)
    pos_weight = torch.tensor([_pos_weight(y[tr], cfg.pos_weight_scale)], device=device)
    loss_fn = F.binary_cross_entropy_with_logits
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = _make_scheduler(opt, cfg, cfg.epochs)

    log_path = LOG_DIR / f"{tag}.jsonl"
    tr_idx = np.flatnonzero(tr)
    va_idx = np.flatnonzero(va)
    best_ap, best_state, best_epoch, bad = -1.0, None, 0, 0
    t0 = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        perm = np.random.permutation(tr_idx)
        model.train()
        epoch_loss, n_batch = 0.0, 0
        for start in range(0, len(perm), cfg.batch_size):
            batch = torch.from_numpy(perm[start : start + cfg.batch_size]).to(device)
            opt.zero_grad()
            logits = model(x_t[batch])
            loss = loss_fn(logits, y_t[batch], pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            epoch_loss += float(loss.item())
            n_batch += 1
        sched.step()

        val_scores = _score_ft(model, x_t, va_idx, batch_size=256)
        val_ap = float(average_precision_score(y[va], val_scores))
        _append_jsonl(log_path, {
            "epoch": epoch,
            "train_loss": round(epoch_loss / max(n_batch, 1), 6),
            "val_ap": round(val_ap, 6),
            "lr": opt.param_groups[0]["lr"],
            "elapsed_s": round(time.perf_counter() - t0, 2),
        })

        if val_ap > best_ap + 1e-6:
            best_ap, best_epoch, bad = val_ap, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics, val_metrics, threshold, ckpt = _finalize(
        y, va, te, best_state, model,
        lambda idx: _score_ft(model, x_t, idx, batch_size=256),
        tag, eval_test, save_ckpt,
    )
    return {
        "model": model_name,
        "setting": setting,
        "split": split,
        "seed": cfg.seed,
        "metrics": metrics,
        "val_metrics": val_metrics,
        "val_ap_best": round(best_ap, 6),
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "params": int(sum(p.numel() for p in model.parameters())),
        "gpu_hours": round((time.perf_counter() - t0) / 3600, 6),
        "checkpoint": ckpt,
    }


def run_experiment(
    model_name, data, structure, split, setting, cfg: TrainConfig,
    device="cuda", masks=None, run_tag=None, eval_test=True, save_ckpt=True,
    model_params=None,
):
    """统一入口：LR/RF 走 sklearn，其余走 torch 训练循环。

    masks=(tr, va, te) 可覆盖默认划分（调参时的 5 折 / 单折验证）。
    """
    kind = "tabular" if model_name in TABULAR + ("FT-Trans",) else "gnn"
    X_raw, need_std = build_feature_matrix(data.x.numpy(), structure, setting, kind)
    if masks is None:
        masks = get_masks(data, split)
    tr, va, te = masks
    X = standardize(X_raw, tr) if need_std else X_raw
    tag = run_tag or f"{model_name}_{setting}_{split}_s{cfg.seed}"

    if model_name in TABULAR:
        from src.models.baselines import run_baseline

        names = [f"f{i}" for i in range(X.shape[1])] if model_name == "RF" else None
        res = run_baseline(
            model_name, X, data.y.numpy(), (tr, va, te), cfg.seed, names, eval_test,
            **(model_params or {}),
        )
        res.update({"setting": setting, "split": split, "seed": cfg.seed})
        return res

    if model_name == "FT-Trans":
        return _train_ft(
            model_name, X, data.y.numpy(), (tr, va, te), cfg, device, tag, split, setting,
            eval_test, save_ckpt,
        )

    return _train_gnn(
        model_name, X, data.y.numpy(), (tr, va, te), data.edge_index, cfg, device, tag,
        split, setting, eval_test, save_ckpt,
    )


def evaluate_checkpoint(model_name, data, structure, split, setting, cfg, ckpt_path, device="cuda"):
    """重新加载 checkpoint，复算验证集 AP（用于可复现性核验）。"""
    kind = "tabular" if model_name == "FT-Trans" else "gnn"
    X_raw, need_std = build_feature_matrix(data.x.numpy(), structure, setting, kind)
    tr, va, _ = get_masks(data, split)
    X = standardize(X_raw, tr) if need_std else X_raw
    n = data.x.shape[0]
    model = build_model(model_name, X.shape[1], cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    x_t = torch.from_numpy(X).to(device)

    if model_name == "FT-Trans":
        scores = _score_ft(model, x_t, np.flatnonzero(va), batch_size=256)
    else:
        if setting in ("feat-only", "local-only"):
            ei = torch.arange(n, device=device).repeat(2, 1)
        else:
            ei = to_undirected(data.edge_index, num_nodes=n).to(device)
        with torch.no_grad():
            logits = model(x_t, ei)[torch.from_numpy(np.flatnonzero(va)).to(device)]
        scores = torch.sigmoid(logits).cpu().numpy()
    return float(average_precision_score(data.y.numpy()[va], scores))
