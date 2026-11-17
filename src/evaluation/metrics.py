"""统一指标：阶梯 AP、ROC-AUC、F1（验证集选阈值）、Precision@K、illicit recall。"""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def best_f1_threshold(y_true, scores) -> float:
    """在验证集上扫描阈值，取 F1 最大处。"""
    y_true = np.asarray(y_true).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return float(thresholds[int(np.argmax(f1[:-1]))])


def compute_metrics(y_true, scores, threshold: float, ks=(100, 500)) -> dict:
    """AP 用 sklearn 的阶梯定义（与 EXP-01 一致），不用梯形 PR-AUC。"""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = scores >= threshold

    tp = int(((pred) & (y_true == 1)).sum())
    fp = int(((pred) & (y_true == 0)).sum())
    fn = int(((~pred) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    order = np.argsort(-scores)
    at_k = {}
    for k in ks:
        at_k[f"p@{k}"] = (
            float(y_true[order[:k]].mean()) if len(y_true) >= k else float("nan")
        )

    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "ap": float(average_precision_score(y_true, scores)),
        "threshold": float(threshold),
        "f1": float(f1),
        "precision": float(precision),
        "recall_illicit": float(recall),
        **at_k,
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }
