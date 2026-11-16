"""按轮次续训 A 赛道基线，并把每轮结果写入 logs/experiments.csv。

核心规则（来自官方续训指南）：
    若平台未采纳某轮候选，应从仍被选中的模型继续。
因此本脚本维护 state.json 记录“当前被采纳的轮次”，父检查点永远取自该轮，
而不是不假思索地接上一轮。指标下降的轮次会被判定为回滚，其检查点保留但不参与后续续训。

用法：
    python scripts/run_round.py --round 1
    python scripts/run_round.py --round 2 --dry-run
    python scripts/run_round.py --round 3 --lr 0.004 --epochs 6
    python scripts/run_round.py --round 4 --accept   强制采纳本轮（需在报告里说明理由）
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROUNDS_PATH = ROOT / "configs" / "rounds.json"
STATE_PATH = ROOT / "state.json"
LOG_PATH = ROOT / "logs" / "experiments.csv"
FIELDS = [
    "round", "stage_id", "parent_round", "parent", "parent_digest_ok",
    "macro_f1_before", "macro_f1_after", "delta", "steps_before", "steps_after",
    "epochs", "learning_rate", "batch_size", "training_seconds", "decision", "note",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_state():
    if not STATE_PATH.exists():
        return {"selected_round": 0, "selected_f1": None,
                "selected_checkpoint": "checkpoint.npz"}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_round(number):
    data = json.loads(ROUNDS_PATH.read_text(encoding="utf-8"))
    for item in data["rounds"]:
        if item["round"] == number:
            merged = dict(data.get("defaults", {}))
            merged.update({k: v for k, v in item.items() if v is not None})
            return data.get("profile", "A"), merged
    raise SystemExit("rounds.json 中没有第 %d 轮" % number)


def resolve_parent(state):
    selected = state["selected_round"]
    if selected == 0:
        return 0, ROOT / "checkpoint.npz"
    return selected, ROOT / "runs" / ("round-%d" % selected) / "checkpoint.npz"


def append_log(row):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="按轮次续训 A 赛道基线（带回滚）")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--accept", action="store_true", help="指标未提升也强制采纳")
    parser.add_argument("--note", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    number = args.round
    profile, config = load_round(number)
    state = load_state()

    parent_round, parent = resolve_parent(state)
    if not parent.exists():
        raise SystemExit("被采纳的检查点 %s 不存在" % parent)

    output = ROOT / "runs" / ("round-%d" % number)
    if output.exists():
        raise SystemExit("输出目录 %s 已存在；官方训练器不允许覆盖历史轮次" % output)

    epochs = args.epochs or config.get("epochs")
    lr = args.lr if args.lr is not None else config.get("learning_rate")
    batch = args.batch or config.get("batch_size")

    cmd = [
        sys.executable, "-m", "baselines.train",
        "--profile", profile,
        "--train", "public/train.jsonl",
        "--validation", "public/validation.jsonl",
        "--from-baseline" if parent_round == 0 else "--resume", str(parent),
        "--epochs", str(epochs),
        "--output", str(output),
        "--stage-id", "round-%d" % number,
    ]
    if lr is not None:
        cmd += ["--learning-rate", str(lr)]
    if batch is not None:
        cmd += ["--batch-size", str(batch)]

    print("本轮假设：" + config.get("hypothesis", "（未填写）"))
    print("本轮改动：" + config.get("change", "（未填写）"))
    print("失败判据：" + config.get("fail_criteria", "（未填写）"))
    print("父检查点：第 %s 轮（%s）" % (
        "基线" if parent_round == 0 else str(parent_round), parent.name))
    print("命令：" + " ".join(cmd))

    if args.dry_run:
        return

    env = dict(os.environ)
    env.setdefault("BASELINE_CPU_THREADS", "1")
    completed = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="strict")
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        raise SystemExit("训练失败，退出码 %d" % completed.returncode)

    report = json.loads((output / "training.json").read_text(encoding="utf-8"))
    digest_ok = report.get("parent_checkpoint_sha256") == sha256_file(parent)
    before = report["validation_before"].get("macro_f1")
    after = report["validation_after"].get("macro_f1")

    reference = state["selected_f1"]
    if reference is None:
        reference = 0.7140297220788709
    improved = after > reference or args.accept
    decision = "采纳" if improved else "回滚"

    if improved:
        save_state({
            "selected_round": number,
            "selected_f1": after,
            "selected_checkpoint": "runs/round-%d/checkpoint.npz" % number,
        })

    append_log({
        "round": number,
        "stage_id": report["stage_id"],
        "parent_round": parent_round,
        "parent": parent.name,
        "parent_digest_ok": digest_ok,
        "macro_f1_before": "%.6f" % before,
        "macro_f1_after": "%.6f" % after,
        "delta": "%+.6f" % (after - before),
        "steps_before": report["initial_steps"],
        "steps_after": report["final_steps"],
        "epochs": report["final_epochs"] - report["initial_epochs"],
        "learning_rate": report["learning_rate"],
        "batch_size": report["batch_size"],
        "training_seconds": "%.4f" % report["training_seconds"],
        "decision": decision,
        "note": args.note,
    })

    print("父检查点校验：" + ("通过" if digest_ok else "不一致，请检查"))
    print("Macro-F1：%.6f -> %.6f（%+.6f）" % (before, after, after - before))
    print("累计步数：%d -> %d，耗时 %.4f 秒" %
          (report["initial_steps"], report["final_steps"], report["training_seconds"]))
    print("决策：%s（参照当前采纳值 %.6f）" % (decision, reference))
    if decision == "回滚":
        nxt = load_state()["selected_round"]
        print("下一轮将从第 %s 轮继续，本轮目录保留作为失败证据" %
              ("基线" if nxt == 0 else str(nxt)))
    print("记录已追加：" + str(LOG_PATH))


if __name__ == "__main__":
    main()
