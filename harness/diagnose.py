"""诊断归因：把漂移现象归到可处理的几类原因，并给出下一轮的候选动作。

对应规则里的第二环"诊断归因"，输出会作为决策器上下文的一部分，
所以模型层与策略层的改动都有据可依，而不是凭感觉调参。
"""

import json


def _train_val_gap(history):
    """最近几轮的学习率/步数与指标变化，用来判断是否在拟合训练批次。"""
    deltas = [float(row["delta"]) for row in history if row.get("delta")]
    return {
        "recent_deltas": deltas,
        "negative_streak": sum(1 for d in deltas[-(3):] if d < 0),
        "total_rounds": len(history),
    }


def diagnose(report, history, thresholds=None):
    thresholds = thresholds or {}
    feature_threshold = float(thresholds.get("feature_score", 0.5))
    label_threshold = float(thresholds.get("label_tv", 0.15))

    overall = report["overall"]
    worst = report["worst_batches"][0] if report.get("worst_batches") else None
    gap = _train_val_gap(history)

    causes = []
    if overall["feature_score"] >= feature_threshold:
        causes.append({
            "cause": "特征漂移",
            "evidence": "整体特征标准化偏移 %.3f，最大单维 z=%.3f（第 %d 维）" % (
                overall["feature_score"], overall["max_z"], overall["max_z_feature"]),
            "action": "维持冻结的标准化参数，改为加强正则或降低步长",
        })
    if overall["label_tv"] >= label_threshold:
        causes.append({
            "cause": "类先验漂移",
            "evidence": "类别分布总变差 %.3f" % overall["label_tv"],
            "action": "引入类别加权，避免多数类继续挤压少数类召回",
        })
    if gap["negative_streak"] >= 2:
        causes.append({
            "cause": "过拟合训练批次",
            "evidence": "最近 %d 轮里有 %d 轮指标下降" % (
                min(3, len(gap["recent_deltas"])) or 3, gap["negative_streak"]),
            "action": "停止增加步数，回到被采纳检查点并缩短轮内步数",
        })
    if worst and worst["feature_score"] > overall["feature_score"] * 1.5:
        causes.append({
            "cause": "批次局部漂移",
            "evidence": "批次 %s 的偏移 %.3f 高于整体 %.3f" % (
                worst["batch"], worst["feature_score"], overall["feature_score"]),
            "action": "在报告里单独说明该批次，避免为单批次过拟合",
        })

    if not causes:
        causes.append({
            "cause": "未检出显著漂移",
            "evidence": "整体特征偏移 %.3f、类先验总变差 %.3f 均在阈值内" % (
                overall["feature_score"], overall["label_tv"]),
            "action": "可继续在当前点位小步试探",
        })

    return {
        "primary": causes[0]["cause"],
        "causes": causes,
        "metrics": {
            "feature_score": overall["feature_score"],
            "max_z": overall["max_z"],
            "label_tv": overall["label_tv"],
            "negative_streak": gap["negative_streak"],
        },
    }


if __name__ == "__main__":
    from harness.monitor import monitor, reference_stats, REFERENCE

    reference = reference_stats()
    report = monitor(reference, REFERENCE.parent / "validation.jsonl")
    print(json.dumps(diagnose(report, []), ensure_ascii=False, indent=2))
