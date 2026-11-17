"""P2：Elliptic 原始数据完整性校验。

产物：results/p2_data_check.json
"""

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "elliptic_bitcoin_dataset"
OUT = ROOT / "results" / "p2_data_check.json"

EXPECTED_DATA_ROWS = {
    "elliptic_txs_features.csv": 203_769,  # 该文件无表头，203,769 行 = 节点数
    "elliptic_txs_classes.csv": 203_769,
    "elliptic_txs_edgelist.csv": 234_355,
}
# 文档 P2 写的期望值（licit 4,203 / illicit 454）漏了一位数字；
# Elliptic 官方口径为 licit 42,019 / illicit 4,545，三者合计 203,769，与 P3 断言一致。
EXPECTED_LABELS = {"licit": 42_019, "illicit": 4_545, "unknown": 157_205}
DOC_LABELS = {"licit": 4_203, "illicit": 454, "unknown": 157_205}
LABEL_MAP = {"1": "illicit", "2": "licit", "unknown": "unknown"}


def count_lines(path: Path) -> int:
    n = 0
    last = b""
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 24)
            if not chunk:
                break
            n += chunk.count(b"\n")
            last = chunk[-1:]
    if last and last != b"\n":
        n += 1  # 末行无换行符
    return n


def read_header(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return next(csv.reader(f))


def main() -> None:
    report = {"files": {}, "labels": {}, "warnings": []}

    for name, expected in EXPECTED_DATA_ROWS.items():
        path = RAW / name
        lines = count_lines(path)
        header = read_header(path)
        try:
            int(header[0])
            has_header = False
        except ValueError:
            has_header = True
        data_rows = lines - (1 if has_header else 0)
        deviation = abs(data_rows - expected) / expected
        report["files"][name] = {
            "lines": lines,
            "has_header": has_header,
            "data_rows": data_rows,
            "expected_data_rows": expected,
            "deviation_pct": round(deviation * 100, 4),
            "n_columns": len(header),
        }
        if deviation > 0.01:
            report["warnings"].append(
                f"{name}: 数据行数 {data_rows} 与期望 {expected} 偏差 {deviation:.2%} > 1%"
            )

    with (RAW / "elliptic_txs_classes.csv").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        counter = Counter(LABEL_MAP.get(row["class"], row["class"]) for row in reader)
    total = sum(counter.values())
    report["labels"] = {
        "total": total,
        "counts": dict(sorted(counter.items())),
        "share": {k: round(v / total, 6) for k, v in sorted(counter.items())},
        "doc_expected": DOC_LABELS,
        "note": (
            "classes.csv 取值为 1=illicit / 2=licit / unknown；文档 P2 的 454 / 4,203 "
            "疑为笔误，官方口径为 4,545 / 42,019（合计 203,769，与 P3 断言一致）"
        ),
    }
    for label, expected in EXPECTED_LABELS.items():
        actual = counter.get(label, 0)
        deviation = abs(actual - expected) / expected
        if deviation > 0.01:
            report["warnings"].append(
                f"标签 {label}: 实际 {actual} 与期望 {expected} 偏差 {deviation:.2%} > 1%"
            )

    report["passed"] = not report["warnings"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
