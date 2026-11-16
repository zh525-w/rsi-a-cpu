"""Harness 的决策层：把"下一轮改什么"抽象成一个可替换接口。

两种实现：
    PresetDecider —— 离线，按 configs/harness.json 里的预设脚本给决策，用于本地跑通链路。
    LLMDecider    —— 通过 OpenAI 兼容接口调用平台模型（千问），只用标准库，不额外引依赖。

统一返回 dict：{"epochs": int|None, "lr": float|None, "batch": int|None,
                "accept": bool, "note": str}
返回 None 表示循环结束。
"""

import json
import os
import urllib.error
import urllib.request


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

    def __init__(self, plan):
        self._plan = {int(item["round"]): item for item in plan}

    def decide(self, round_number, context):
        item = self._plan.get(round_number)
        if item is None:
            return None
        action = {k: v for k, v in item.items() if k != "round"}
        return action

    def charge(self, prompt_tokens, completion_tokens):
        return 0.0


class LLMDecider:
    """通过 OpenAI 兼容接口让平台模型给出决策。

    需要的环境变量（都不写在代码里）：
        PA_MODEL_BASE_URL   例如 https://xxx/v1
        PA_MODEL_API_KEY    平台下发的 key
        PA_MODEL_NAME       模型名，例如千问 3.8 27B 的平台标识
    """

    name = "llm"

    def __init__(self, base_url, api_key, model_name, price=None, timeout=60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.price = price or {}
        self.timeout = timeout
        self.spent = 0.0

    @classmethod
    def from_env(cls, config):
        base = os.environ.get("PA_MODEL_BASE_URL", "")
        key = os.environ.get("PA_MODEL_API_KEY", "")
        name = os.environ.get("PA_MODEL_NAME", "")
        if not (base and key and name):
            return None
        return cls(base, key, name, config.get("price"))

    def decide(self, round_number, context):
        payload = {
            "model": self.model_name,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(
                    {"round": round_number, "history": context}, ensure_ascii=False)},
            ],
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.api_key},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise DecisionError("模型接口返回 %d：%s" % (error.code, error.read()[:200]))
        except urllib.error.URLError as error:
            raise DecisionError("模型接口不可达：%s" % error.reason)

        usage = body.get("usage", {})
        self.spent += self.charge(usage.get("prompt_tokens", 0),
                                  usage.get("completion_tokens", 0))

        content = body["choices"][0]["message"]["content"].strip()
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
