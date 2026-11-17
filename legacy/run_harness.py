"""Harness 入口：按平台契约分派 describe / train / predict 三个模式。

    python scripts/run_harness.py describe     # 只读，输出能力清单（JSON）
    python scripts/run_harness.py train        # 五环闭环（可加 --unattended / --decider llm）
    python scripts/run_harness.py predict      # 独立预测入口（读 /data/test.jsonl 写 /output）

不带子命令时等价于 train，兼容镜像里原有的 CMD ["--unattended"]。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODES = ("describe", "train", "predict")


def main():
    argv = sys.argv[1:]
    mode = argv[0] if argv and argv[0] in MODES else "train"
    rest = argv[1:] if argv and argv[0] in MODES else argv

    from harness import modes

    if mode == "describe":
        import argparse
        import json
        parser = argparse.ArgumentParser(description="输出工作区能力清单")
        parser.add_argument("--max-rounds", type=int, default=None)
        args = parser.parse_args(rest)
        config_rounds = None
        config_path = ROOT / "configs" / "harness.json"
        if config_path.is_file():
            config_rounds = json.loads(config_path.read_text(encoding="utf-8")).get("max_rounds")
        modes.describe(args.max_rounds, config_rounds)
        return 0

    if mode == "predict":
        import argparse
        parser = argparse.ArgumentParser(description="独立预测模式")
        parser.add_argument("--input", default=None, help="默认读 /data/test.jsonl")
        parser.add_argument("--output", default=None, help="默认写 /output/predictions.jsonl")
        parser.add_argument("--checkpoint", default=None, help="默认读 /model/checkpoint.npz")
        parser.add_argument("--format", choices=["official", "pairs"], default="official",
                            help="official=官方协议 outputs=[{label,score}]；pairs=[[label,score]]")
        parser.add_argument("--limit", type=int, default=None)
        args = parser.parse_args(rest)
        modes.predict(args.input, args.output, args.checkpoint, args.format, args.limit)
        return 0

    return modes.train(rest)


if __name__ == "__main__":
    sys.exit(main() or 0)
