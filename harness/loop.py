"""Harness 主循环：漂移监测 → 诊断归因 → 实验优化 → 评估上线 → 回滚。

规则要求系统在封闭评测中无人工干预地跑完全部阶段，因此本文件是 train 模式的实现：
    python scripts/run_harness.py train --unattended      # 平台契约入口
    python scripts/run_harness.py --unattended            # 本地等价写法

受控机制（防止无限制迭代）：
    轮次上限（平台下发 > 环境变量 > 命令行 > 配置 > 10）、
    累计时长上限 max_wall_seconds、Token 上限 token_budget、
    连续回滚上限 stop_after_rollback_streak。
任何一条触发即收尾，并在 results/final.json 里写明停止原因与最终产物。
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from harness import paths
from harness.decide import DecisionError, LLMDecider, PresetDecider, validate
from harness.diagnose import diagnose
from harness.monitor import monitor, reference_stats
from harness.swan import connect, finish, log_round

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "harness.json"
BASELINE_F1 = 0.7140297220788709


def log_path():
    return paths.logs_dir() / "harness.jsonl"


def ledger_path():
    return paths.logs_dir() / "experiments.csv"


def read_history():
    path = ledger_path()
    if not path.exists():
        return []
    import csv
    return list(csv.DictReader(open(path, encoding="utf-8")))


def read_state():
    path = paths.state_path()
    if not path.exists():
        return {"selected_round": 0, "selected_f1": None,
                "selected_checkpoint": "checkpoint.npz"}
    return json.loads(path.read_text(encoding="utf-8"))


def append_record(record):
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_command(round_number, action, dry_run):
    cmd = [sys.executable, "scripts/run_round.py", "--round", str(round_number)]
    if action["epochs"] is not None:
        cmd += ["--epochs", str(action["epochs"])]
    if action["lr"] is not None:
        cmd += ["--lr", str(action["lr"])]
    if action["batch"] is not None:
        cmd += ["--batch", str(action["batch"])]
    if action["accept"]:
        cmd += ["--accept"]
    if action["note"]:
        cmd += ["--note", action["note"]]
    if dry_run:
        cmd += ["--dry-run"]
    return cmd


def make_decider(config, kind):
    if kind == "llm":
        decider, note = LLMDecider.from_env(config)
        if decider is None:
            raise SystemExit("没有可用的大模型 provider：%s\n"
                             "正式评测需平台注入 model() 网关；本地可用 PA_MODEL_* 环境变量，"
                             "或用 --decider preset 离线跑通链路" % note)
        if note:
            print("模型 provider：%s（%s）" % (decider.provider, note))
        return decider
    return PresetDecider(config.get("preset_plan", []))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Harness 五环闭环（train 模式）")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--decider", choices=["preset", "llm"], default=None)
    parser.add_argument("--unattended", action="store_true", help="无人值守：跑满预算或收敛后退出")
    args = parser.parse_args(argv)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    max_rounds = paths.rounds_limit(args.max_rounds, config.get("max_rounds"))
    max_wall = float(config.get("max_wall_seconds", 0))
    token_budget = float(config.get("token_budget", 0))
    rollback_limit = int(config.get("stop_after_rollback_streak", 3))
    decider = make_decider(config, args.decider or config.get("decider", "preset"))
    swan = connect(config)

    print("=== 漂移监测 ===")
    reference = reference_stats()
    drift = monitor(reference, paths.validation_path())
    print("参照集 %d 行，验证流 %d 行 / %d 个批次" % (
        drift["reference_size"], drift["validation_size"], drift["batches"]))
    print("整体特征偏移 %.3f，最大单维 z=%.3f，类先验总变差 %.3f" % (
        drift["overall"]["feature_score"], drift["overall"]["max_z"],
        drift["overall"]["label_tv"]))
    for item in drift["worst_batches"]:
        print("  批次 %s：偏移 %.3f（%d 行）" % (
            item["batch"], item["feature_score"], item["n"]))

    print("\n=== 诊断归因 ===")
    history = read_history()
    diagnosis = diagnose(drift, history, config.get("thresholds"))
    for cause in diagnosis["causes"]:
        print("- %s：%s → %s" % (cause["cause"], cause["evidence"], cause["action"]))

    start_round = len(history) + 1
    spent_tokens = 0
    spent_cost = 0.0
    started_at = time.time()
    rolled_back = 0
    stop_reason = "已达轮次上限"

    print("\n=== 实验优化 / 评估上线 / 回滚 ===")
    print("决策器=%s，从第 %d 轮起，上限 %d 轮（来源：%s）%s" % (
        decider.name, start_round, max_rounds,
        paths.rounds_source(args.max_rounds, config.get("max_rounds")),
        "（dry-run）" if args.dry_run else ""))

    for offset in range(max_rounds):
        round_number = start_round + offset
        elapsed_total = time.time() - started_at
        if max_wall and elapsed_total > max_wall:
            stop_reason = "累计时长超过上限"
            break
        if token_budget and spent_tokens >= token_budget:
            stop_reason = "Token 预算用尽"
            break
        if rolled_back >= rollback_limit:
            stop_reason = "连续回滚达到上限，停止以免无限制迭代"
            break

        context = {
            "selected_round": read_state()["selected_round"],
            "selected_f1": read_state()["selected_f1"],
            "baseline_f1": BASELINE_F1,
            "diagnosis": diagnosis,
            "history": read_history()[-4:],
            "rounds_left": max_rounds - offset,
        }

        round_tokens = {"prompt_tokens": 0, "completion_tokens": 0}
        try:
            action = validate(decider.decide(round_number, context))
        except DecisionError as error:
            append_record({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "round": round_number,
                           "decider": decider.name, "status": "决策被拒", "reason": str(error)})
            print("第 %d 轮决策被拒：%s" % (round_number, error))
            stop_reason = "决策被拒"
            break
        finally:
            usage = getattr(decider, "last_usage", {}) or {}
            round_tokens = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            }
            spent_tokens += round_tokens["prompt_tokens"] + round_tokens["completion_tokens"]
            spent_cost += decider.charge(round_tokens["prompt_tokens"],
                                         round_tokens["completion_tokens"])

        if action is None:
            print("决策器未给出第 %d 轮方案，认为收敛" % round_number)
            append_record({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "round": round_number,
                           "decider": decider.name, "status": "收敛停止"})
            stop_reason = "决策器收敛"
            break

        cmd = build_command(round_number, action, args.dry_run)
        print("\n--- 第 %d 轮 ---" % round_number)
        print("决策：" + json.dumps(action, ensure_ascii=False))

        round_started = time.time()
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                                   text=True, encoding="utf-8", errors="replace")
        seconds = round(time.time() - round_started, 3)

        if completed.returncode != 0:
            append_record({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "round": round_number,
                           "decider": decider.name, "status": "执行失败", "action": action,
                           "wall_seconds": seconds,
                           "stderr_tail": completed.stderr.strip().splitlines()[-3:]})
            print("执行失败：%s" % completed.stderr.strip().splitlines()[-1:])
            stop_reason = "执行失败"
            break

        history = read_history()
        last = history[-1] if history else {}
        deployed = (not args.dry_run) and last.get("decision") == "采纳"
        if not args.dry_run and not deployed:
            rolled_back += 1
        elif deployed:
            rolled_back = 0

        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "round": round_number,
            "decider": decider.name,
            "provider": getattr(decider, "provider", "preset"),
            "stage": ["漂移监测", "诊断归因", "实验优化", "评估上线", "回滚"],
            "diagnosis_primary": diagnosis["primary"],
            "action": action,
            "status": "dry-run" if args.dry_run else "已执行",
            "wall_seconds": seconds,
            "macro_f1_after": None if args.dry_run else last.get("macro_f1_after"),
            "decision": None if args.dry_run else last.get("decision"),
            "deployed": deployed,
            "prompt_tokens": round_tokens["prompt_tokens"],
            "completion_tokens": round_tokens["completion_tokens"],
            "cost_cny": decider.charge(round_tokens["prompt_tokens"],
                                       round_tokens["completion_tokens"]),
        }
        append_record(record)
        if swan is not None and not args.dry_run:
            log_round(swan, record)
        if not args.dry_run:
            print("结果：Macro-F1=%s，决策=%s，Token=%d" % (
                record["macro_f1_after"], record["decision"],
                record["prompt_tokens"] + record["completion_tokens"]))
            diagnosis = diagnose(monitor(reference, paths.validation_path()),
                                 history, config.get("thresholds"))

    state = read_state()
    best_f1 = state.get("selected_f1") or BASELINE_F1
    results = paths.results_dir()
    results.mkdir(parents=True, exist_ok=True)
    final = {
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stop_reason": stop_reason,
        "decider": decider.name,
        "model_provider": getattr(decider, "provider", "preset"),
        "rounds_limit": max_rounds,
        "rounds_source": paths.rounds_source(args.max_rounds, config.get("max_rounds")),
        "rounds_run": len(read_history()) if not args.dry_run else 0,
        "selected_round": state.get("selected_round"),
        "selected_checkpoint": state.get("selected_checkpoint"),
        "final_macro_f1": best_f1,
        "baseline_macro_f1": BASELINE_F1,
        "meets_static_baseline": best_f1 >= BASELINE_F1,
        "total_wall_seconds": round(time.time() - started_at, 3),
        "total_tokens": int(spent_tokens),
        "total_cost_cny": round(spent_cost, 6),
        "dry_run": bool(args.dry_run),
    }
    (results / "final.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finish(swan, final)

    print("\n=== 收尾 ===")
    print("停止原因：%s" % stop_reason)
    print("最终检查点：%s（Macro-F1 %.6f）" % (
        final["selected_checkpoint"], best_f1))
    print("静态基线：%.6f，门槛%s" % (
        BASELINE_F1, "满足" if final["meets_static_baseline"] else "未满足"))
    print("Token 合计 %d，成本 %.6f 元" % (final["total_tokens"], final["total_cost_cny"]))
    print("循环记录：%s" % log_path())
    print("结果摘要：%s" % (results / "final.json"))


if __name__ == "__main__":
    main()
