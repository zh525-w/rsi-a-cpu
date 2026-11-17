"""漂移监测：以初始训练集为参照，量化新批次的特征偏移与类先验偏移。

对应规则里的第一环"漂移监测"。参照统计量固定取自 public/train.jsonl，
与"标准化参数只由初始训练集拟合、不得重估"这条硬边界一致。
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "public" / "train.jsonl"


def load_rows(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def reference_stats(path=None):
    path = Path(path or REFERENCE)
    features, labels = [], []
    for row in load_rows(path):
        features.append(row["inputs"]["features"])
        labels.append(row["answer"])
    matrix = np.asarray(features, dtype=np.float64)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-9] = 1e-9
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    total = sum(counts.values())
    prior = {k: v / total for k, v in counts.items()}
    return {"mean": mean, "std": std, "prior": prior, "n": len(labels)}


def _label_prior(labels):
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def _tv_distance(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def batch_drift(reference, rows):
    """单批次相对参照统计量的偏移。

    返回 feature_score（各特征标准化均值偏移的绝对值均值）、
    max_z（最大单特征偏移）、top_features（偏移最大的若干维）、label_tv（类先验总变差）。
    """
    matrix = np.asarray([row["inputs"]["features"] for row in rows], dtype=np.float64)
    mean = matrix.mean(axis=0)
    z = (mean - reference["mean"]) / reference["std"]
    abs_z = np.abs(z)
    top = np.argsort(abs_z)[::-1][:5]
    prior = _label_prior([row["answer"] for row in rows])
    return {
        "n": len(rows),
        "feature_score": float(abs_z.mean()),
        "max_z": float(abs_z.max()),
        "top_features": [int(i) for i in top],
        "max_z_feature": int(np.argmax(abs_z)),
        "label_tv": float(_tv_distance(prior, reference["prior"])),
    }


def monitor(reference, validation_path):
    """对整个验证流做一次监测，并按批次列出偏移最大的几组。"""
    rows = list(load_rows(validation_path))
    overall = batch_drift(reference, rows)

    grouped = {}
    for row in rows:
        batch = row.get("meta", {}).get("source_batch")
        grouped.setdefault(batch, []).append(row)

    per_batch = []
    for batch, group in grouped.items():
        item = batch_drift(reference, group)
        item["batch"] = batch
        per_batch.append(item)
    per_batch.sort(key=lambda x: x["feature_score"], reverse=True)

    return {
        "reference_size": reference["n"],
        "validation_size": len(rows),
        "batches": len(grouped),
        "overall": overall,
        "worst_batches": per_batch[:3],
    }


if __name__ == "__main__":
    reference = reference_stats()
    report = monitor(reference, ROOT / "public" / "validation.jsonl")
    print(json.dumps(report, ensure_ascii=False, indent=2))
