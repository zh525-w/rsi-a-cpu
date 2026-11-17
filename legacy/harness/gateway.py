"""模型调用层：把「驱动 Harness 的大模型」抽象成可替换的 provider。

两个 provider：
    PlatformGateway   —— 正式评测使用。平台在离线容器内注入 model() 网关，
                         外部 HTTP 不可达，因此必须走这条。
    OpenAICompat      —— 本地/联调使用。OpenAI 兼容的 /chat/completions，
                         仅作为降级方案，不能用于正式提交。

契约 contest.harness.workspace.v2 里尚未随镜像下发的部分（网关模块名、函数签名、
Token 用量字段）集中在本文件的探测逻辑里：探测失败时抛出 GatewayUnavailable，
并把试过的候选项原样写进错误信息。宁可报错，也不静默退回「假装调用成功」。
"""

import importlib
import json
import os
import urllib.error
import urllib.request


class GatewayUnavailable(Exception):
    pass


CANDIDATE_MODULES = ("harness_runtime", "workspace_runtime", "platform_runtime",
                     "model_gateway", "harness_gateway", "rsi_runtime")
CANDIDATE_ATTRS = ("model", "chat", "complete", "invoke", "generate")


class ChatResult:
    def __init__(self, content, prompt_tokens=0, completion_tokens=0, provider="unknown"):
        self.content = content
        self.prompt_tokens = int(prompt_tokens or 0)
        self.completion_tokens = int(completion_tokens or 0)
        self.provider = provider


def _extract(resp, provider):
    """把网关返回体归一化成 ChatResult；兼容常见几种形状。"""
    if isinstance(resp, str):
        return ChatResult(resp, provider=provider)
    if not isinstance(resp, dict):
        raise GatewayUnavailable("%s 返回了无法识别的类型 %r" % (provider, type(resp)))

    usage = resp.get("usage") or {}
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))

    content = None
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message") or {}
            content = message.get("content") if isinstance(message, dict) else None
            if content is None:
                content = first.get("text")
    for key in ("content", "text", "output", "response"):
        if content is None:
            content = resp.get(key)
    if content is None:
        raise GatewayUnavailable("%s 返回体缺少文本字段：%s" % (
            provider, json.dumps(resp, ensure_ascii=False)[:200]))
    return ChatResult(content, prompt, completion, provider)


class PlatformGateway:
    """平台注入的 model() 网关。

    探测顺序：模块（可由 HARNESS_GATEWAY_MODULES 覆盖）→ 属性名 → 调用签名。
    任何一步失败都抛 GatewayUnavailable，错误信息里带上试过的全部候选项。
    """

    provider = "platform_gateway"

    def __init__(self, callable_obj, module_name, model_name=None):
        self._call = callable_obj
        self.module_name = module_name
        self.model_name = model_name or os.environ.get("PA_MODEL_NAME", "")
        self.attempted = []

    @classmethod
    def discover(cls, model_name=None):
        modules = [m.strip() for m in os.environ.get("HARNESS_GATEWAY_MODULES", "").split(",") if m.strip()]
        modules = modules or list(CANDIDATE_MODULES)
        tried = []
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                tried.append("%s（无法导入）" % module_name)
                continue
            for attr in CANDIDATE_ATTRS:
                candidate = getattr(module, attr, None)
                if callable(candidate):
                    return cls(candidate, "%s.%s" % (module_name, attr), model_name)
            tried.append("%s（无 %s）" % (module_name, "/".join(CANDIDATE_ATTRS)))
        raise GatewayUnavailable(
            "未找到平台注入的 model() 网关，已尝试：%s。"
            "如契约确认了模块名，用 HARNESS_GATEWAY_MODULES 指定" % "；".join(tried))

    def chat(self, system_prompt, user_prompt):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        attempts = (
            lambda: self._call(messages=messages, model=self.model_name),
            lambda: self._call(messages=messages),
            lambda: self._call(messages, self.model_name),
            lambda: self._call(messages),
            lambda: self._call(prompt=user_prompt, system=system_prompt, model=self.model_name),
        )
        last_error = None
        for index, attempt in enumerate(attempts, 1):
            try:
                return _extract(attempt(), self.provider)
            except TypeError as error:
                last_error = error
                self.attempted.append("signature#%d" % index)
                continue
        raise GatewayUnavailable(
            "%s 调用签名不匹配，已尝试 %d 种写法：%s" % (
                self.module_name, len(attempts), last_error))


class OpenAICompat:
    """OpenAI 兼容接口（仅本地/联调）。正式评测环境无外网，不要依赖它。"""

    provider = "openai_compat"

    def __init__(self, base_url, api_key, model_name, timeout=60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout

    @classmethod
    def from_env(cls):
        base = os.environ.get("PA_MODEL_BASE_URL", "")
        key = os.environ.get("PA_MODEL_API_KEY", "")
        name = os.environ.get("PA_MODEL_NAME", "")
        if not (base and key and name):
            return None
        return cls(base, key, name)

    def chat(self, system_prompt, user_prompt):
        payload = {
            "model": self.model_name,
            "temperature": 0,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}],
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
            raise GatewayUnavailable("模型接口返回 %d：%s" % (error.code, error.read()[:200]))
        except urllib.error.URLError as error:
            raise GatewayUnavailable("模型接口不可达：%s" % error.reason)
        result = _extract(body, self.provider)
        result.content = result.content.strip()
        return result


def build_client(config=None, prefer=None):
    """按「平台网关优先、OpenAI 兼容兜底」选出唯一 provider。

    返回 (client, provider_name, note)：
        client  为 None 表示没有任何可用 provider（调用方应明确失败，不得伪造）。
    """
    config = config or {}
    prefer = (prefer or config.get("model_provider") or "auto").strip()
    note = ""
    if prefer in ("auto", "platform_gateway"):
        try:
            return PlatformGateway.discover(), PlatformGateway.provider, "平台注入网关"
        except GatewayUnavailable as error:
            if prefer == "platform_gateway":
                return None, prefer, str(error)
            note = "平台网关不可用（%s）；回落到本地兼容接口" % error
    if prefer in ("auto", "openai_compat"):
        client = OpenAICompat.from_env()
        if client is not None:
            return client, OpenAICompat.provider, note or "本地 OpenAI 兼容端点"
        return None, prefer, (note + "；" if note else "") + "未配置 PA_MODEL_* 环境变量"
    return None, prefer, "未知 provider：%s" % prefer
