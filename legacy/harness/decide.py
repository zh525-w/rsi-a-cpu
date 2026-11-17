"""Harness 的决策层：把"下一轮改什么"抽象成一个可替换接口。

两种实现：
    PresetDecider —— 离线，按 configs/harness.json 里的预设脚本给决策，用于本地跑通链路。
    LLMDecider    —— 调用驱动大模型（正式评测走平台 model() 网关，本地可走 OpenAI 兼容端点），
                     只用标准库，不额外引依赖。

Token 用量由 provider 返回、逐轮写进台账，不再固定写 0——否则「大模型 Token」这一项
会被判成没调用大模型，与规则任务 2 直接冲突。
"""

import json

from harness.gateway import GatewayUnavailable, build_client


class DecisionError(Exception):
    pass


def validate(action):
    if action is None:
        return None
    if not isinstance(action, dict):
        raise DecisionError("决策必须是对象，收到 %r" % type(action))

    allowed = {"epochs", "lr", "batch", "accept", "note"}
    unknown = set(action) - allowed
    if unknown:
        raise DecisionError("决策里出现未支持的键：%s" % "、".join(sorted(unknown)))

    epochs = action.get("epochs")
    if epochs is not None and not (1 <= int(epochs) <= 50):
        raise DecisionError("epochs 必须在 1..50，收到 %r" % epochs)

    lr = action.get("lr")
    if lr is not None and not (1e-5 <= float(lr) <= 0.1):
        raise DecisionError("lr 必须在 1e-5..0.1，收到 %r" % lr)

    batch = action.get("batch")
    if batch is not None and not (8 <= int(batch) <= 512):
        raise DecisionError("batch 必须在 8..512，收到 %r" % batch)

    return {
        "epochs": None if epochs is None else int(epochs),
        "lr": None if lr is None else float(lr),
        "batch": None if batch is None else int(batch),
        "accept": bool(action.get("accept", False)),
        "note": str(action.get("note", "")),
    }


class PresetDecider:
    """离线决策：按预设脚本逐轮下发，用尽即停止。"""

    name = "preset"
    provider = "preset"

    def __init__(self, plan):
        self._plan = {int(item["round"]): item for item in plan}
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "provider": self.provider}
        self.note = "离线预设脚本，不调用大模型"

    def decide(self, round_number, context):
        item = self._plan.get(round_number)
        if item is None:
            return None
        action = {k: v for k, v in item.items() if k != "round"}
        return action

    def charge(self, prompt_tokens, completion_tokens):
        return 0.0


class LLMDecider:
    """让大模型给出每轮决策。

    provider 由 harness.gateway 决定：正式评测用平台注入的 model() 网关，
    本地调试可退到 PA_MODEL_BASE_URL 指向的 OpenAI 兼容端点。
    """

    name = "llm"

    def __init__(self, client, price=None, provider=None, note=""):
        self.client = client
        self.price = price or {}
        self.provider = provider or getattr(client, "provider", "unknown")
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "provider": self.provider}
        self.note = note

    @classmethod
    def from_env(cls, config):
        """返回 (decider, note)；没有可用 provider 时 decider 为 None，note 说明原因。"""
        try:
            client, provider, note = build_client(config)
        except GatewayUnavailable as error:
            return None, str(error)
        if client is None:
            return None, note
        return cls(client, config.get("price"), provider, note), note

    def decide(self, round_number, context):
        body = json.dumps({"round": round_number, "history": context}, ensure_ascii=False)
        try:
            result = self.client.chat(SYSTEM_PROMPT, body)
        except GatewayUnavailable as error:
            raise DecisionError("模型调用失败：%s" % error)

        self.last_usage = {
            "prompt_tokens": getattr(result, "prompt_tokens", 0),
            "completion_tokens": getattr(result, "completion_tokens", 0),
            "provider": getattr(result, "provider", self.provider),
        }

        content = (result.content or "").strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            content = content[4:] if content.startswith("json") else content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            raise DecisionError("模型输出不是 JSON：%s" % content[:200])

    def charge(self, prompt_tokens, completion_tokens):
        return round(
            prompt_tokens / 1000.0 * float(self.price.get("prompt_per_1k", 0.0))
            + completion_tokens / 1000.0 * float(self.price.get("completion_per_1k", 0.0)), 6)


SYSTEM_PROMPT = (
    "你是模型训练的实验决策器。每一轮只能改一个变量，"
    "并且必须遵守三条硬边界：标准化参数不得重估、验证集不得并入训练、"
    "批次不得随机混合切分。只输出 JSON，字段仅限 "
    "epochs / lr / batch / accept / note，未改的字段留空。"
)
