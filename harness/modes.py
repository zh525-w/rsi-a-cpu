"""三个工作区模式：describe / train / predict。

平台契约要求入口按模式分派（`describe` 报能力、`train` 跑闭环、`predict` 出预测），
本文件是这三个模式的实现，`scripts/run_harness.py` 只做分派。

约定：
    describe  只读，输出一份 JSON 能力清单，不产生任何写入；
    train     交给 harness.loop，保持原有「监测→诊断→决策→训练评估→采纳/回滚」五环；
    predict   独立预测入口，不参与训练、不读标签，逐行输出预测结果。
"""

import json
import sys
import time
from pathlib import Path

from harness import paths


MANIFEST_PATH = paths.ROOT / "adapter_manifest.json"


def load_manifest():
    if not MANIFEST_PATH.is_file():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def checkpoint_meta():
    """读取检查点结构信息；失败时返回 None，不让 describe 因此崩掉。"""
    try:
        from baselines.model import Model
        model = Model.load(str(paths.checkpoint_path()))
        return {
            "task": model.meta.get("task"),
            "classes": model.meta.get("classes"),
            "parameter_count": model.meta.get("parameter_count"),
            "feature_count": len(model.meta.get("paths", [])),
        }
    except Exception as error:  # 描述模式必须能跑起来，哪怕检查点缺失
        return {"error": "%s: %s" % (type(error).__name__, error)}


def describe(rounds_cli=None, config_rounds=None):
    manifest = load_manifest()
    meta = checkpoint_meta()
    body = {
        "schema": manifest.get("schema", "contest.harness.workspace.v2"),
        "modes": ["describe", "train", "predict"],
        "task": {
            "track": "A",
            "name": "气体传感器漂移下的六类气体识别",
            "metric": "macro_f1",
            "classes": meta.get("classes"),
            "feature_count": meta.get("feature_count"),
            "parameter_count": meta.get("parameter_count"),
        },
        "mounts": paths.describe_mounts(),
        "data": {
            "train": str(paths.train_path()) if paths.train_path().is_file() else None,
            "validation": str(paths.validation_path()) if paths.validation_path().is_file() else None,
            "test": str(paths.test_path()) if paths.test_path() else None,
        },
        "checkpoint": str(paths.checkpoint_path()),
        "rounds": paths.rounds_limit(rounds_cli, config_rounds),
        "rounds_source": paths.rounds_source(rounds_cli, config_rounds),
        "adapter_source_ref": manifest.get("adapter_source_ref"),
        "entry": {
            "describe": "python scripts/run_harness.py describe",
            "train": "python scripts/run_harness.py train",
            "predict": "python scripts/run_harness.py predict",
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(json.dumps(body, ensure_ascii=False, indent=2))
    return body


def predict(input_path=None, output_path=None, checkpoint=None, fmt="official", limit=None):
    """逐行读输入、输出预测。

    默认走官方预测协议（与基线包 `baselines/predict.py` 一致）：
    输入每行 `{"task": ..., "inputs": ...}`，输出每行 `{"outputs": [{"label": ..., "score": ...}]}`。
    `--input -` 时从 stdin 逐行读、逐行写 stdout，可被平台按管道调用。
    """
    try:
        from baselines.model import Model
        from baselines.predict import predict as predict_one
    except ImportError as error:
        raise SystemExit("无法导入 baselines 预测代码：%s" % error)

    ckpt = Path(checkpoint) if checkpoint else paths.checkpoint_path()
    if not ckpt.is_file():
        raise SystemExit("检查点不存在：%s" % ckpt)
    model = Model.load(str(ckpt))

    use_stdin = input_path == "-"
    handle = None
    if use_stdin:
        reader = sys.stdin
        dst = Path(output_path) if output_path else None
    else:
        src = Path(input_path) if input_path else paths.test_path()
        if src is None or not Path(src).is_file():
            raise SystemExit("未找到预测输入：/data 下没有 test.jsonl，请用 --input 指定，"
                             "或用 --input - 从 stdin 读")
        handle = open(src, "r", encoding="utf-8")
        reader = handle
        dst = Path(output_path) if output_path else (paths.output_dir() / "predictions.jsonl")
        dst.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    try:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            if not isinstance(request, dict) or "inputs" not in request:
                raise SystemExit("预测请求必须包含 inputs 字段：%s" % line[:120])
            if request.get("task") and request["task"] != model.meta["task"]:
                raise SystemExit("预测任务不匹配：%s" % request["task"])
            outputs = predict_one(model, request["inputs"])
            if fmt == "official":
                result = outputs
            elif isinstance(outputs.get("outputs"), list):
                result = [[item["label"], item["score"]] for item in outputs["outputs"]]
            else:
                result = [[outputs["outputs"]["value"]]]
            payload = json.dumps(result, ensure_ascii=False, allow_nan=False)
            if dst is None:
                print(payload, flush=True)
            else:
                with open(dst, "a", encoding="utf-8") as writer:
                    writer.write(payload + "\n")
            rows += 1
            if limit and rows >= limit:
                break
    finally:
        if handle is not None:
            handle.close()

    print("预测完成：%d 行 → %s" % (rows, dst if dst is not None else "stdout"))
    return rows


def train(argv):
    from harness.loop import main as loop_main
    return loop_main(argv)
