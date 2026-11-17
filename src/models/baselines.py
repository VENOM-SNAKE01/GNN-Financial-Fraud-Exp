"""LR / RF 基线：训练集标准化、验证集选 F1 阈值、测试集报指标。"""

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import best_f1_threshold, compute_metrics


def run_baseline(
    name: str, X, y, masks, seed: int, feature_names=None, eval_test=True,
    C: float = 1.0, n_estimators: int = 500, max_depth=None,
    min_samples_leaf: int = 1, max_features="sqrt",
) -> dict:
    tr, va, te = masks
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr])
    Xva = scaler.transform(X[va])
    Xte = scaler.transform(X[te])

    if name == "LR":
        clf = LogisticRegression(
            C=C, class_weight="balanced", max_iter=5000, random_state=seed
        )
    elif name == "RF":
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    else:
        raise ValueError(name)

    t0 = time.perf_counter()
    clf.fit(Xtr, y[tr])
    train_seconds = time.perf_counter() - t0

    va_scores = clf.predict_proba(Xva)[:, 1]
    threshold = best_f1_threshold(y[va], va_scores)
    val_metrics = compute_metrics(y[va], va_scores, threshold)
    metrics = None
    if eval_test:
        te_scores = clf.predict_proba(Xte)[:, 1]
        metrics = compute_metrics(y[te], te_scores, threshold)

    result = {
        "model": name,
        "metrics": metrics,
        "val_metrics": val_metrics,
        "train_seconds": round(train_seconds, 2),
        "gpu_hours": 0.0,
        "params": None,
    }
    if name == "RF" and feature_names is not None:
        imp = clf.feature_importances_
        order = np.argsort(-imp)[:20]
        result["feature_importance_top20"] = [
            {"feature": feature_names[i], "importance": float(imp[i])} for i in order
        ]
    return result
