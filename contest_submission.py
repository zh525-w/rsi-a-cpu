"""智训引擎 RSI CHALLENGE 提交入口（contest.harness.workspace.v2）。

平台每轮启动 `python3 /app/harness.py train`，把一帧 start 写进 stdin；
SDK 解帧后调用本文件的 train(rnd)。一轮只做一件事：读取父检查点，
在公开训练集上续训一小步，用一次真实模型调用决定超参，提交候选并结束本轮。
预测走 `python3 /app/harness.py predict`，逐行读 stdin、逐行写 stdout。

三个数据集共用一套代码：
- 特征按“路径”抽取，A 的扁平 features、B 的双轴嵌套特征、C 的 settings+sensors 都能量化；
- 任务类型取 rnd.task，B/C 是回归（RUL），A 是分类；
- 回归先归一化目标再训练，预测时反归一化并取非负，与官方 baseline 口径一致。
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
TRAIN_CAP = 24000
VALIDATION_CAP = 30000


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


def load_rows(iterator, cap, seed=42, label="train"):
    """流式读取并做定长水库抽样：C 有 11 万行，全量入内存会顶到镜像内存上限。"""
    rng = np.random.default_rng(seed)
    rows = []
    seen = 0
    for row in iterator:
        seen += 1
        if len(rows) < cap:
            rows.append(row)
        else:
            slot = int(rng.integers(0, seen))
            if slot < cap:
                rows[slot] = row
    log("%s：读到 %d 行，保留 %d 行" % (label, seen, len(rows)))
    return rows


def matrix(rows, paths):
    result = np.zeros((len(rows), len(paths)), dtype=np.float64)
    for index, row in enumerate(rows):
        result[index] = core.extract(row.get("inputs") or {}, paths)
    return result


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
    rows = load_rows(rnd.data.training_rows(), TRAIN_CAP, seed=42, label="训练集")
    validation = load_rows(rnd.data.validation_rows(), VALIDATION_CAP, seed=43, label="验证集")
    log("dataset=%s round=%d task=%s metric=%s(%s) rows=%d validation=%d"
        % (rnd.dataset_key, rnd.round_index, kind, metric, direction, len(rows), len(validation)))
    if not rows:
        raise RuntimeError("公开训练集为空")

    parent_net, parent_meta, parent_name = load_parent(rnd)
    paths = parse_paths(parent_meta.get("paths")) if parent_meta else []
    if not paths or not usable(paths, rows):
        paths = core.union_paths(rows[:200], core.MAX_FEATURES)
        log("父模型未给出可用路径，按本数据集 inputs 结构生成 %d 维特征" % len(paths))

    x = matrix(rows, paths)
    x_validation = matrix(validation, paths)

    if kind == "regression":
        answers = np.asarray([float(row.get("answer", 0.0)) for row in rows], dtype=np.float64)
        answers_validation = np.asarray(
            [float(row.get("answer", 0.0)) for row in validation], dtype=np.float64)
        classes = []
        target_mean = float(answers.mean())
        target_std = float(answers.std())
        target_scale = target_std if target_std > 1e-9 else 1.0
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
            [lookup.get(str(row.get("answer")), 0) for row in validation], dtype=np.int64)
        target_mean, target_scale = 0.0, 1.0

    mean, scale = core.fit_scaler(x)
    inherited = can_inherit(parent_net, x, kind, classes, paths)
    if inherited:
        mean, scale = parent_net.mean, parent_net.scale
        if kind == "regression":
            target_mean, target_scale = parent_net.target_mean, parent_net.target_scale

    if inherited:
        net = parent_net
        log("继承父模型 w1%s，参数 %d 个" % (tuple(net.w1.shape), core_parameter_count(net)))
    else:
        net = core.Net(x.shape[1], out_dim=(len(classes) if kind == "classification" else 1),
                       kind=kind, seed=42)
        log("父模型与当前数据集不兼容，按 %d 维初始化新模型" % x.shape[1])
    net.mean, net.scale = mean, scale
    net.classes = classes
    net.paths = paths
    net.target_mean, net.target_scale = target_mean, target_scale

    parent_score = (core.score(net, x_validation, answers_validation, classes, kind, metric)
                    if (inherited and len(validation)) else None)
    log("父模型验证分 %s" % fmt(parent_score))

    context = (
        "数据集 %s，第 %d 轮；任务 %s，指标 %s（%s）；特征 %d 维；训练 %d 行，验证 %d 行；"
        "父模型当前验证分 %s。目标是在不超过 %s 秒内小幅续训并尽量提升该指标。"
        % (rnd.dataset_key, rnd.round_index, kind, metric, direction, x.shape[1],
           len(rows), len(validation), fmt(parent_score), rnd.wall_seconds or 1800)
    )
    receipt, choice = ask_decider(rnd, context)

    # 逐轮配方表：lr、epochs、batch 一起换，避免同一配方重复跑出同一个候选
    schedule = (
        (0.0018, 2, 128), (0.0030, 3, 64), (0.0007, 6, 64), (0.0020, 3, 64),
        (0.0040, 3, 96), (0.0030, 5, 128), (0.0012, 4, 64), (0.0030, 2, 64),
        (0.0007, 2, 128), (0.0030, 3, 128),
    )
    slot = schedule[(int(rnd.round_index) - 1) % len(schedule)]
    choice["learning_rate"], choice["epochs"], choice["batch_size"] = slot
    log("第 %d 轮配方 lr=%s epochs=%s batch=%s" % (int(rnd.round_index), *slot))

    net.fit(x, answers, epochs=choice["epochs"], learning_rate=choice["learning_rate"],
            batch_size=int(choice.get("batch_size", 128)), seed=42)
    candidate_score = (core.score(net, x_validation, answers_validation, classes, kind, metric)
                       if len(validation) else None)
    log("候选验证分 %s（lr=%s epochs=%s）" % (fmt(candidate_score), choice["learning_rate"], choice["epochs"]))
    origin = "继承父权重续训" if inherited else "从零初始化"

    # 回归轮做双路探针：B 的官方权重在验证轴承上已过拟合，续训越训越差，
    # 从零重训反而更好；C 则相反。两路各训一版，按验证分择优提交。
    if kind == "regression" and inherited and len(validation):
        probe_seed = 42 + 7 * int(rnd.round_index)          # 每轮换一个起点，避免重复同一版本
        probe_epochs = choice["epochs"] + (int(rnd.round_index) % 3)
        probe = core.Net(x.shape[1], out_dim=1, kind="regression", seed=probe_seed)
        probe.mean, probe.scale = mean, scale
        probe.paths = paths
        probe.target_mean, probe.target_scale = target_mean, target_scale
        probe.fit(x, answers, epochs=probe_epochs, learning_rate=choice["learning_rate"],
                  batch_size=128, seed=probe_seed)
        log("探针配置 seed=%d epochs=%d" % (probe_seed, probe_epochs))
        probe_score = core.score(probe, x_validation, answers_validation, classes, kind, metric)
        log("双路探针：从零初始化验证分 %s" % fmt(probe_score))
        if better(probe_score, candidate_score, direction):
            log("从零初始化的版本更优，改用该版本")
            net, candidate_score, origin = probe, probe_score, "从零初始化（父权重在本数据集验证集上过拟合）"

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
        origin)
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

    state_dir = unique_dir(work, "state-r%d" % rnd.round_index)
    state_dir = rnd.state_workspace(state_dir)
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
    vector = core.extract(sample.inputs or {}, paths)
    matrix_in = net.normalize(vector[None, :])
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
    candidates = [name for name in names if name.endswith(".npz")]
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


def better(left, right, direction):
    if left is None:
        return False
    if right is None:
        return True
    if direction == "lower":
        return left < right - TOLERANCE
    return left > right + TOLERANCE


def rollback_streak(rnd):
    """统计连续“保留父模型”的轮数，用于自动衰减学习率。"""
    count = 0
    try:
        for item in reversed(list(rnd.history())):
            config = item.get("configuration") or {}
            if str(config.get("selected")) == "parent":
                count += 1
            else:
                break
    except Exception:  # history 不可用时按 0 处理
        return 0
    return count


def is_improved(parent_score, candidate_score, direction):
    if candidate_score is None:
        return False
    if parent_score is None:
        return True
    if direction == "lower":
        return candidate_score <= parent_score - TOLERANCE
    return candidate_score >= parent_score + TOLERANCE
