"""平台挂载点解析：优先读 /data /previous /model /output，本地目录兜底。

平台契约（contest.harness.workspace.v2）在正式评测时会把四个目录挂进容器：
    /data      公开训练与验证数据、隐藏测试输入
    /previous  上一轮检查点与状态
    /model     初始模型/基线权重
    /output    本轮产物与日志
本地开发时这四个目录不存在，自动回落到仓库根，保证同一条命令在两边都能跑。

每个路径都允许用环境变量覆盖（HARNESS_*_DIR），便于在没有契约原文时先做联调。
不做任何猜测式拼接：目录不存在就明确回落，绝不静默创建一个假的挂载点。
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_MOUNTS = {
    "data": ("HARNESS_DATA_DIR", "/data"),
    "previous": ("HARNESS_PREVIOUS_DIR", "/previous"),
    "model": ("HARNESS_MODEL_DIR", "/model"),
    "output": ("HARNESS_OUTPUT_DIR", "/output"),
}

_SUITE_FILENAMES = ("workspace_suite.json", "suite.json", "workspace.json", "config.json")


def mount(name):
    """返回真实存在的挂载点；不存在返回 None（不创建、不猜测）。"""
    env_name, default = _MOUNTS[name]
    candidate = Path(os.environ.get(env_name) or default)
    try:
        return candidate if candidate.is_dir() else None
    except OSError:
        return None


def data_dir():
    return mount("data") or (ROOT / "public")


def previous_dir():
    return mount("previous") or ROOT


def model_dir():
    return mount("model") or ROOT


def output_dir():
    return mount("output") or ROOT


def train_path():
    return data_dir() / "train.jsonl"


def validation_path():
    return data_dir() / "validation.jsonl"


def test_path():
    """隐藏测试输入；本地没有该文件时返回 None。"""
    for name in ("test.jsonl", "hidden.jsonl", "test.json"):
        candidate = data_dir() / name
        if candidate.is_file():
            return candidate
    return None


def runs_dir():
    return output_dir() / "runs"


def logs_dir():
    return output_dir() / "logs"


def results_dir():
    return output_dir() / "results"


def state_path():
    return output_dir() / "state.json"


def checkpoint_path():
    """续训起点：优先平台 /model 下发的检查点，其次仓库内的初始检查点。"""
    for candidate in (model_dir() / "checkpoint.npz", previous_dir() / "checkpoint.npz",
                      ROOT / "checkpoint.npz"):
        if candidate.is_file():
            return candidate
    return ROOT / "checkpoint.npz"


def _dig(obj, key):
    """在嵌套结构里找第一个同名键（平台下发的 suite 结构层级未定）。"""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _dig(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _dig(value, key)
            if found is not None:
                return found
    return None


def suite_config():
    """读取平台下发的 workspace_suite；没有则返回空 dict。"""
    candidates = []
    env_suite = os.environ.get("HARNESS_SUITE")
    if env_suite:
        candidates.append(Path(env_suite))
    for base in (mount("data"), mount("previous"), mount("output")):
        if base:
            for name in _SUITE_FILENAMES:
                candidates.append(base / name)
    for path in candidates:
        try:
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return {}


def rounds_limit(cli_rounds=None, config_rounds=None, fallback=10):
    """轮次优先级：平台下发 > 环境变量 > 命令行 > 配置 > 缺省 10。

    平台规则里轮次由 workspace_suite.rounds 决定（缺省 10），本队配置里的 8
    只在本地调试时生效，不再写死进镜像。
    """
    suite_rounds = _dig(suite_config(), "rounds")
    env_rounds = os.environ.get("HARNESS_ROUNDS")
    for value in (suite_rounds, env_rounds, cli_rounds, config_rounds, fallback):
        if value in (None, ""):
            continue
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return fallback


def rounds_source(cli_rounds=None, config_rounds=None):
    """给 describe 用的溯源标签。"""
    suite_rounds = _dig(suite_config(), "rounds")
    env_rounds = os.environ.get("HARNESS_ROUNDS")
    if suite_rounds not in (None, ""):
        return "workspace_suite"
    if env_rounds not in (None, ""):
        return "env:HARNESS_ROUNDS"
    if cli_rounds not in (None, ""):
        return "cli:--max-rounds"
    if config_rounds not in (None, ""):
        return "config:harness.json"
    return "fallback:10"


def describe_mounts():
    return {name: (str(mount(name)) if mount(name) else None) for name in _MOUNTS}
