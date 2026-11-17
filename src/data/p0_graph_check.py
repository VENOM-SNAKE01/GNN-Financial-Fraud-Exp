"""P0：Elliptic 图结构前提检查（go / no-go）。

产物：
  results/p0_graph_check.json
  results/figures/p0_degree_distribution.png
  results/figures/p0_component_sizes.png
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, ScalarFormatter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "elliptic_bitcoin_dataset"
OUT_JSON = ROOT / "results" / "p0_graph_check.json"
FIG_DIR = ROOT / "results" / "figures"

LABEL_MAP = {"1": "illicit", "2": "licit"}

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def plain_log_ticks(ax) -> None:
    """对数轴用普通数字刻度，避免 mathtext 负号缺字形。"""
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())


def save_fig(fig, path: Path) -> None:
    tmp = path.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(path)


def main() -> None:
    feats = pd.read_csv(
        RAW / "elliptic_txs_features.csv", header=None, usecols=[0, 1], names=["txId", "time_step"]
    )
    classes = pd.read_csv(RAW / "elliptic_txs_classes.csv", dtype={"class": str})
    edges = pd.read_csv(RAW / "elliptic_txs_edgelist.csv")

    node_ids = feats["txId"].to_numpy()
    n = len(node_ids)
    idx = pd.Series(np.arange(n), index=node_ids)

    assert classes["txId"].isin(idx.index).all(), "classes 中存在 features 未覆盖的节点"
    assert edges["txId1"].isin(idx.index).all() and edges["txId2"].isin(idx.index).all()

    label = np.array(["unknown"] * n, dtype=object)
    cls = classes.set_index("txId")["class"]
    mapped = cls.map(LABEL_MAP)
    for tid, val in mapped.items():
        if isinstance(val, str):
            label[idx[tid]] = val

    src = idx[edges["txId1"]].to_numpy()
    dst = idx[edges["txId2"]].to_numpy()
    m = len(src)

    # 无向度（每条边对两端各计 1）
    degree = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)
    # 唯一无向邻居数
    adj_sets = [set() for _ in range(n)]
    for a, b in zip(src, dst):
        adj_sets[a].add(b)
        adj_sets[b].add(a)
    uniq_deg = np.array([len(s) for s in adj_sets])

    ill = label == "illicit"
    lic = label == "licit"
    labeled = ill | lic
    n_ill, n_lic = int(ill.sum()), int(lic.sum())

    # 连通分量（无向）
    graph = csr_matrix((np.ones(m), (src, dst)), shape=(n, n))
    n_comp, comp = connected_components(graph, directed=False)
    comp_sizes = np.bincount(comp)

    # 分量与时间步的关系：Elliptic 的边只连接同一时间步内的交易
    ts = feats["time_step"].to_numpy()
    comp_ts = pd.crosstab(comp, ts).gt(0).sum(axis=1)
    ts_coverage = []
    for t in np.unique(ts):
        nodes_t = np.flatnonzero(ts == t)
        comps_t = comp[nodes_t]
        uniq, cnt = np.unique(comps_t, return_counts=True)
        big = uniq[cnt.argmax()]
        ill_t = nodes_t[ill[nodes_t]]
        if len(ill_t):
            ts_coverage.append(float((comp[ill_t] == big).mean()))

    # 1 跳邻居标签纯度
    baseline = n_ill / (n_ill + n_lic)
    ill_neigh_links = 0
    ill_neigh_ill = 0
    ill_per_node_purity = []
    ill_no_labeled = 0
    lic_neigh_links = 0
    lic_neigh_ill = 0
    for i in np.flatnonzero(ill):
        nb = np.fromiter(adj_sets[i], dtype=np.int64)
        nb_lab = label[nb][label[nb] != "unknown"]
        if len(nb_lab) == 0:
            ill_no_labeled += 1
            ill_per_node_purity.append(np.nan)
            continue
        k = int((nb_lab == "illicit").sum())
        ill_neigh_ill += k
        ill_neigh_links += len(nb_lab)
        ill_per_node_purity.append(k / len(nb_lab))
    for i in np.flatnonzero(lic):
        nb = np.fromiter(adj_sets[i], dtype=np.int64)
        nb_lab = label[nb][label[nb] != "unknown"]
        if len(nb_lab) == 0:
            continue
        lic_neigh_ill += int((nb_lab == "illicit").sum())
        lic_neigh_links += len(nb_lab)

    purity_pooled = ill_neigh_ill / ill_neigh_links if ill_neigh_links else float("nan")
    purity_arr = np.array(ill_per_node_purity, dtype=float)
    purity_valid = purity_arr[~np.isnan(purity_arr)]
    binom = binomtest(ill_neigh_ill, ill_neigh_links, p=baseline, alternative="greater")
    licit_purity = lic_neigh_ill / lic_neigh_links if lic_neigh_links else float("nan")

    # 2 跳邻域规模（唯一节点数，不含自身）
    def two_hop(i: int) -> int:
        one = adj_sets[i]
        two = set()
        for v in one:
            two |= adj_sets[v]
        two |= one
        two.discard(i)
        return len(two)

    hop2_ill = np.array([two_hop(int(i)) for i in np.flatnonzero(ill)])
    hop2_lic = np.array([two_hop(int(i)) for i in np.flatnonzero(lic)])

    med_ill_deg = float(np.median(degree[ill]))
    med_lic_deg = float(np.median(degree[lic]))
    isolated_ratio = float((degree == 0).mean())
    med2_ill, med2_lic = float(np.median(hop2_ill)), float(np.median(hop2_lic))
    ratio2 = max(med2_ill, med2_lic) / max(min(med2_ill, med2_lic), 1e-9)

    checks = {
        "illicit_count": {
            "value": n_ill,
            "criterion": "文档期望约 454（官方口径 4,545）；判据：<100 视为标注异常",
            "passed": n_ill >= 100,
        },
        "illicit_median_degree": {
            "value": med_ill_deg,
            "criterion": ">= 2",
            "passed": med_ill_deg >= 2,
            "blocking": False,
            "decision": "用户裁定（2026-11-17）：如实记录、不作阻断；正类多为 1 度叶节点，但全图孤立节点为 0",
            "note": (
                f"正类中度数为 1 的占 {float((degree[ill] == 1).mean()):.1%}，"
                f"但全图孤立节点为 {int((degree == 0).sum())} 个"
            ),
        },
        "isolated_ratio": {
            "value": round(isolated_ratio, 6),
            "criterion": "全图 <= 5%（>5% 需记录并评估影响）",
            "passed": isolated_ratio <= 0.05,
        },
        "illicit_in_timestep_largest_component": {
            "value": round(float(np.mean(ts_coverage)), 6),
            "criterion": ">= 0.95（按所属时间步内最大分量；原全图口径不适用）",
            "passed": float(np.mean(ts_coverage)) >= 0.95,
            "note": (
                f"全图恰好 {n_comp} 个连通分量、无跨时间步的边（最大分量仅占全图 "
                f"{comp_sizes.max() / n:.2%}），故按时间步口径评估；用户已确认（2026-11-17）"
            ),
        },
        "illicit_neighbor_purity": {
            "pooled_purity": round(purity_pooled, 6),
            "baseline_illicit_rate": round(baseline, 6),
            "lift": round(purity_pooled / baseline, 4),
            "binom_p_value": float(binom.pvalue),
            "criterion": "显著高于全图 illicit 基线率",
            "passed": bool(purity_pooled > baseline and binom.pvalue < 1e-3),
        },
        "two_hop_size_ratio": {
            "median_illicit": med2_ill,
            "median_licit": med2_lic,
            "ratio": round(ratio2, 4),
            "criterion": "< 3",
            "passed": ratio2 < 3,
        },
    }

    report = {
        "dataset": {
            "nodes": n,
            "edges": m,
            "labeled": {"illicit": n_ill, "licit": n_lic, "total": int(labeled.sum())},
            "time_step_range": [int(feats["time_step"].min()), int(feats["time_step"].max())],
        },
        "graph": {
            "components": int(n_comp),
            "components_spanning_multiple_time_steps": int((comp_ts > 1).sum()),
            "largest_component_size": int(comp_sizes.max()),
            "largest_component_share": round(float(comp_sizes.max() / n), 6),
            "illicit_in_own_timestep_largest_component": round(float(np.mean(ts_coverage)), 6),
            "isolated_nodes": int((degree == 0).sum()),
            "degree": {
                "all": {
                    "mean": round(float(degree.mean()), 4),
                    "median": float(np.median(degree)),
                    "p95": float(np.percentile(degree, 95)),
                    "max": int(degree.max()),
                },
                "illicit": {
                    "mean": round(float(degree[ill].mean()), 4),
                    "median": med_ill_deg,
                    "p95": float(np.percentile(degree[ill], 95)),
                    "isolated_ratio": round(float((degree[ill] == 0).mean()), 6),
                },
                "licit": {
                    "mean": round(float(degree[lic].mean()), 4),
                    "median": med_lic_deg,
                    "p95": float(np.percentile(degree[lic], 95)),
                    "isolated_ratio": round(float((degree[lic] == 0).mean()), 6),
                },
                "unique_neighbor_median_illicit": float(np.median(uniq_deg[ill])),
                "unique_neighbor_median_licit": float(np.median(uniq_deg[lic])),
            },
        },
        "neighbor_purity": {
            "illicit_pooled_purity": round(purity_pooled, 6),
            "illicit_per_node_mean": round(float(purity_valid.mean()), 6),
            "illicit_per_node_median": round(float(np.median(purity_valid)), 6),
            "illicit_with_labeled_neighbor_ratio": round(1 - ill_no_labeled / n_ill, 6),
            "illicit_labeled_neighbor_links": ill_neigh_links,
            "licit_neighbor_illicit_share": round(licit_purity, 6),
            "baseline_illicit_rate": round(baseline, 6),
        },
        "two_hop": {
            "illicit": {
                "mean": round(float(hop2_ill.mean()), 4),
                "median": med2_ill,
                "p95": float(np.percentile(hop2_ill, 95)),
            },
            "licit": {
                "mean": round(float(hop2_lic.mean()), 4),
                "median": med2_lic,
                "p95": float(np.percentile(hop2_lic, 95)),
            },
            "median_ratio": round(ratio2, 4),
        },
        "checks": checks,
        "passed": all(c["passed"] for c in checks.values() if c.get("blocking", True)),
    }

    if report["passed"]:
        report["conclusion_zh"] = (
            f"正类 {n_ill} 个，1 跳已标注邻居中 illicit 占 {purity_pooled:.1%}，"
            f"为基线 {baseline:.1%} 的 {purity_pooled / baseline:.1f} 倍（p={binom.pvalue:.1e}）；"
            f"2 跳邻域中位数正/负类 {med2_ill:g} vs {med2_lic:g}（{ratio2:.2f} 倍）；"
            f"全图 0 孤立节点，{n_comp} 个分量对应 {int(ts.max())} 个时间步，"
            f"正类在所属时间步最大分量内覆盖 {np.mean(ts_coverage):.0%}。"
            f"结构携带信号，可进入建模；正类度数中位数 {med_ill_deg:g} 已如实记录。"
        )
    else:
        failed = [k for k, c in checks.items() if not c["passed"] and c.get("blocking", True)]
        report["conclusion_zh"] = (
            f"未通过阻断性检查项：{'、'.join(failed)}。结构信号不足，应停止实验并汇报。"
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(degree[ill], bins=np.arange(0, 41), alpha=0.6, label="illicit",
                 density=True, color="#c0392b")
    axes[0].hist(degree[lic], bins=np.arange(0, 41), alpha=0.5, label="licit",
                 density=True, color="#2980b9")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("无向度数")
    axes[0].set_ylabel("密度（对数）")
    axes[0].set_title("已标注节点度数分布（0–40）")
    axes[0].legend()
    axes[1].hist(degree[ill], bins=np.arange(0, 41), alpha=0.7, label="illicit",
                 color="#c0392b")
    axes[1].hist(degree[lic], bins=np.arange(0, 41), alpha=0.5, label="licit",
                 color="#2980b9")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("无向度数")
    axes[1].set_ylabel("节点数（对数）")
    axes[1].set_title("正负类度数计数对比")
    axes[1].legend()
    plain_log_ticks(axes[0])
    plain_log_ticks(axes[1])
    save_fig(fig, FIG_DIR / "p0_degree_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(np.sort(comp_sizes)[::-1], marker=".", linestyle="none", markersize=3)
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("连通分量排名")
    axes[0].set_ylabel("分量规模（对数）")
    axes[0].set_title(f"连通分量规模（共 {n_comp} 个，最大 {int(comp_sizes.max())}）")
    vals = [purity_pooled, purity_valid.mean(), np.median(purity_valid), baseline]
    names = ["正类邻居\n池化纯度", "正类\n逐节点均值", "正类\n逐节点中位数", "全图违规\n基线率"]
    bars = axes[1].bar(names, vals, color=["#c0392b", "#e67e22", "#f1c40f", "#7f8c8d"])
    axes[1].set_ylabel("illicit 占比")
    axes[1].set_title("1 跳已标注邻居标签纯度 vs 基线")
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center")
    axes[1].set_ylim(0, max(vals) * 1.25)
    plain_log_ticks(axes[0])
    save_fig(fig, FIG_DIR / "p0_component_sizes.png")

    print(json.dumps({k: report[k] for k in ["dataset", "checks", "passed", "conclusion_zh"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
