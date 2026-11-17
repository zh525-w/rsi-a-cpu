"""智训引擎 RSI CHALLENGE 提交入口（contest.harness.workspace.v2）。

平台每轮启动 `python3 /app/harness.py train`，把一帧 start 写进 stdin；
SDK 解帧后调用本文件的 train(rnd)。一轮只做一件事：读取父检查点，
在公开训练集上续训一小步，用一次真实模型调用决定超参，提交候选并结束本轮。
预测走 `python3 /app/harness.py predict`，逐行读 stdin、逐行写 stdout。
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from contest_harness import Round

import rsi.core as core

TASKS = ("classification", "regression")
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
DEFAULTS = {"learning_rate": 0.003, "epochs": 3}
TOLERANCE = 1e-6


# ------------------------------------------------------------------ 小工具
def log(message):
    """日志一律走 stderr，stdout 只留给协议帧。"""
    print("[rsi] " + str(message), file=sys.stderr, flush=True)


def task_kind(contract):
    if isinstance(contract, dict):
        for key in ("task", "task_type", "type", "objective", "kind"):
            value = contract.get(key)
            if value in ("classification", "regression"):
                return value
        for value in contract.values():
            if isinstance(value, str) and value in ("classification", "regression"):
                return value
    return "classification"


def metric_info(contract):
    name, direction = "macro_f1", "higher"
    if isinstance(contract, dict):
        raw = contract.get("metric")
        if isinstance(raw, str):
            name = raw
        elif isinstance(raw, dict):
            name = str(raw.get("name") or raw.get("metric") or name)
        if contract.get("metric_direction") in ("higher", "lower"):
            direction = contract["metric_direction"]
    return name.lower(), direction


def work_root():
    return Path(os.environ.get("CONTEST_WORK") or "/workspace")


def unique_dir(base, name):
    target = base / name
    suffix = 1
    while target.exists():
        suffix += 1
        target = base / ("%s-%d" % (name, suffix))
    return target


def clip(text, limit=3900):
    """SDK 按字节卡 4000 上限，中文按字符截断会超，这里按字节收。"""
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", "ignore")


def parse_paths(raw):
    """官方 metadata 里的 paths 每项是 JSON 字符串，这里统一成路径列表。"""
    parsed = []
    for item in raw or []:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                return []
        if not isinstance(item, list):
            return []
        path = []
        for step in item:
            if not isinstance(step, list) or len(step) != 2:
                return []
            if step[0] not in ("key", "index"):
                return []
            path.append([step[0], str(step[1]) if step[0] == "key" else int(step[1])])
        parsed.append(path)
    return parsed


def ask_decider(rnd, context):
    """每轮至少一次真实模型调用；解析失败时退回默认超参，调用号照样记入 finish。"""
    messages = [
        {"role": "system", "content": "你是工业小模型的训练决策器。只回复一行 JSON，不要解释。"},
        {"role": "user", "content": context + "\n请给出本轮的学习率和训练轮数，格式：{\"learning_rate\":0.003,\"epochs\":3}"},
    ]
    receipt = rnd.model(messages, max_tokens=min(96, rnd.max_output_tokens), temperature=0)
    text = receipt.text or ""
    log("decider call=%s reply=%s" % (receipt.call_id, text.strip()[:200]))
    match = re.search(r"\{.{0,200}?\}", text, re.S)
    chosen = dict(DEFAULTS)
    if match:
        try:
            payload = json.loads(match.group(0))
            rate = float(payload.get("learning_rate", DEFAULTS["learning_rate"]))
            epochs = int(payload.get("epochs", DEFAULTS["epochs"]))
            chosen["learning_rate"] = min(max(rate, 1e-4), 0.02)
            chosen["epochs"] = min(max(epochs, 1), 8)
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    return receipt, chosen


# ------------------------------------------------------------------ 训练一轮
def train(rnd: Round):
    kind = task_kind(rnd.task)
    metric, direction = metric_info(rnd.task)
    rows = list(rnd.data.training_rows())
    validation = list(rnd.data.validation_rows())
    log("dataset=%s round=%d task=%s metric=%s rows=%d validation=%d"
        % (rnd.dataset_key, rnd.round_index, kind, metric, len(rows), len(validation)))
    if not rows:
        raise RuntimeError("公开训练集为空")

    # 特征路径：优先沿用父检查点声明的路径（官方 baseline 就是 128 维扁平 features）
    parent_net, parent_meta, parent_name = load_parent(rnd)
    paths = parse_paths(parent_meta.get("paths")) if parent_meta else []
    if not paths or not usable(paths, rows):
        paths = core.union_paths([row for row in rows[:64]], core.MAX_FEATURES)
        log("父模型未给出可用路径，按本数据集 inputs 结构生成 %d 维特征" % len(paths))

    x = core.matrix(rows, paths)
    x_validation = core.matrix(validation, paths)

    if kind == "regression":
        answers = np.asarray([float(row.get("answer", 0.0)) for row in rows], dtype=np.float64)
        answers_validation = np.asarray([float(row.get("answer", 0.0)) for row in validation], dtype=np.float64)
        classes = []
    else:
        labels = sorted({str(row.get("answer")) for row in rows})
        parent_classes = list(parent_net.classes) if parent_net is not None else []
        if parent_classes and len(parent_classes) == len(labels) and set(parent_classes) == set(labels):
            classes = parent_classes
            log("沿用父检查点的类别顺序 %s" % classes)
        else:
            classes = labels
        lookup = {label: index for index, label in enumerate(classes)}
        answers = np.asarray([lookup[str(row.get("answer"))] for row in rows], dtype=np.int64)
        answers_validation = np.asarray(
            [lookup.get(str(row.get("answer")), 0) for row in validation], dtype=np.int64
        )

    mean, scale = core.fit_scaler(x)
    if parent_net is not None and parent_net.w1.shape[0] == x.shape[1]:
        mean, scale = parent_net.mean, parent_net.scale
    x_normalized = (x - mean) / scale
    x_validation_normalized = (x_validation - mean) / scale

    inherited = can_inherit(parent_net, x, kind, classes, paths)
    if inherited:
        net = parent_net
        log("继承父模型 w1%s 参数 %d 个" % (tuple(net.w1.shape), core_parameter_count(net)))
    else:
        net = core.Net(x.shape[1], out_dim=(len(classes) if kind == "classification" else 1),
                       kind=kind, seed=42)
        log("父模型与当前数据集不兼容，按 %d 维初始化新模型" % x.shape[1])
    net.mean, net.scale = mean, scale
    net.classes = classes
    net.paths = paths

    parent_score = (core.score_rows(net, x_validation_normalized, answers_validation, classes, kind)
                    if (inherited and validation) else None)
    log("父模型验证分 %s" % parent_score)

    context = (
        "数据集 %s，第 %d 轮；任务 %s，指标 %s（%s）；特征 %d 维；训练 %d 行，验证 %d 行；"
        "父模型当前验证分 %s。目标是在不超过 %s 秒内小幅续训并尽量提升该指标。"
        % (rnd.dataset_key, rnd.round_index, kind, metric, direction, x.shape[1],
           len(rows), len(validation), parent_score, rnd.wall_seconds or 1800)
    )
    receipt, choice = ask_decider(rnd, context)

    net.fit(x_normalized, answers, epochs=choice["epochs"], learning_rate=choice["learning_rate"], seed=42)
    candidate_score = core.score_rows(net, x_validation_normalized, answers_validation, classes, kind) if validation else None
    log("候选验证分 %s（lr=%s epochs=%s）" % (candidate_score, choice["learning_rate"], choice["epochs"]))

    improved = is_improved(parent_score, candidate_score, direction)

    work = work_root()
    candidate_dir = unique_dir(work, "candidate-r%d" % rnd.round_index)
    candidate_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = candidate_dir / "checkpoint.npz"
    net.save(checkpoint, extra={
        "dataset_key": rnd.dataset_key,
        "round_index": rnd.round_index,
        "metric": metric,
        "parent_score": parent_score,
        "candidate_score": candidate_score,
        "learning_rate": choice["learning_rate"],
        "epochs": choice["epochs"],
        "model_call_id": receipt.call_id,
    })

    hypothesis = "%s：在第 %d 轮把学习率取 %.4g、续训 %d 轮，验证分应从 %s 改善。" % (
        metric, rnd.round_index, choice["learning_rate"], choice["epochs"], fmt(parent_score))
    change_summary = "从父检查点 %s 继续 Adam 训练 %d 轮（lr=%.4g），特征 %d 维，%s。" % (
        parent_name, choice["epochs"], choice["learning_rate"], x.shape[1],
        "参数维度一致，直接继承父权重" if inherited else "维度不一致，按当前数据集重建权重")
    submitted = rnd.submit_candidate(
        candidate_dir,
        hypothesis=clip(hypothesis),
        change_summary=clip(change_summary),
        configuration={
            "learning_rate": choice["learning_rate"],
            "epochs": choice["epochs"],
            "features": int(x.shape[1]),
            "parent_score": parent_score,
            "candidate_score": candidate_score,
            "metric": metric,
        },
        seed=42,
        checkpoint_paths=["checkpoint.npz"],
        code_paths=[],
        predict_argv=PREDICT_ARGV,
    )
    log("候选已提交 artifact=%s" % submitted.get("artifact_id"))

    state_dir = rnd.state_workspace(work / ("state-r%d" % rnd.round_index))
    (state_dir / "round.json").write_text(json.dumps({
        "dataset_key": rnd.dataset_key,
        "round_index": rnd.round_index,
        "metric": metric,
        "parent_score": parent_score,
        "candidate_score": candidate_score,
        "selected": "candidate" if improved else "parent",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    rnd.save_state(state_dir)

    history = rnd.history()
    selected = submitted["artifact_id"] if improved else rnd.parent.artifact_id
    reason = "候选验证分 %s，父模型 %s，%s；模型调用 %s。" % (
        fmt(candidate_score), fmt(parent_score),
        "采纳候选" if improved else "保留父模型", receipt.call_id)
    rnd.finish(
        selected_artifact_id=selected,
        hypothesis=clip(hypothesis),
        change_summary=clip(change_summary),
        reason=clip(reason),
        configuration={
            "learning_rate": choice["learning_rate"],
            "epochs": choice["epochs"],
            "features": int(x.shape[1]),
            "parent_score": parent_score,
            "candidate_score": candidate_score,
            "metric": metric,
            "selected": "candidate" if improved else "parent",
        },
        seed=42,
        history_ids=[item["id"] for item in history],
        model_call_ids=list(rnd.call_ids),
    )


# ------------------------------------------------------------------ 预测
def predict(sample, model_dir):
    checkpoint = os.environ.get("CONTEST_ARTIFACT_PATH") or str(Path(model_dir) / "checkpoint.npz")
    net, _meta = core.Net.load(checkpoint)
    paths = net.paths or core.leaf_paths(sample.inputs)
    vector = core.extract(sample.inputs, paths)
    matrix_in = ((vector - net.mean) / net.scale)[None, :]
    if sample.task == "regression" or net.kind == "regression":
        value = float(net.predict(matrix_in)[0])
        return {"value": value if np.isfinite(value) else 0.0}
    prob = net.predict(matrix_in)[0]
    classes = net.classes or ["normal"]
    outputs = [{"label": str(label), "score": float(round(float(prob[index]), 6))}
               for index, label in enumerate(classes[:len(prob)])]
    top_index = max(range(len(outputs)), key=lambda index: outputs[index]["score"])
    top_score = outputs[top_index]["score"]
    for index, item in enumerate(outputs):
        if index != top_index and item["score"] >= top_score:
            item["score"] = max(0.0, round(top_score - 1e-6, 9))
    return outputs


# ------------------------------------------------------------------ 内部助手
def load_parent(rnd):
    names = list(rnd.parent.declared_paths())
    candidates = [name for name in names if name.endswith(".npz")] or []
    for name in candidates:
        try:
            path = rnd.parent.path(name)
            net, meta = core.Net.load(path)
            net.paths = parse_paths(meta.get("paths")) or net.paths
            log("读取父检查点 %s（参数 %d 个，特征 %d 维）"
                % (name, core_parameter_count(net), len(net.paths)))
            return net, meta, name
        except Exception as error:  # 父检查点损坏或不是我们的格式
            log("父检查点 %s 无法载入：%s" % (name, error))
    return None, {}, "无（本轮为首轮或父检查点不可读）"


def usable(paths, rows):
    for row in rows[:8]:
        if all(core.value_at(row.get("inputs"), path) is not None for path in paths):
            return True
    return False


def can_inherit(net, x, kind, classes, paths):
    if net is None:
        return False
    if net.kind != kind or net.w1.shape[0] != x.shape[1]:
        return False
    if net.paths and json.dumps(net.paths, sort_keys=True) != json.dumps(paths, sort_keys=True):
        return False
    if kind == "classification":
        return len(net.classes) > 0 and len(net.classes) == len(classes)
    return net.w2.shape[1] == 1


def core_parameter_count(net):
    return int(net.w1.size + net.b1.size + net.w2.size + net.b2.size)


def fmt(value):
    if value is None:
        return "N/A"
    return "%.6f" % value


def is_improved(parent_score, candidate_score, direction):
    if candidate_score is None:
        return False
    if parent_score is None:
        return True
    if direction == "lower":
        return candidate_score <= parent_score - TOLERANCE
    return candidate_score >= parent_score + TOLERANCE
