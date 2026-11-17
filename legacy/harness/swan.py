"""SwanLab 接入：把每轮指标写进云端实验记录。

设计原则是"接了不影响本地闭环，没接不影响结果产出"：
未安装、未启用或云端失败时只打印提示，循环继续，本地日志照常落盘。

云端模式需要账号，先设环境变量：
    SWANLAB_API_KEY=xxx
再在 configs/harness.json 里把 swanlab.mode 改成 cloud。
本地自测可用 mode=local，产物落在工程目录下的 swanlab/ 里，不需要登录。
"""

import time


def connect(config):
    options = (config or {}).get("swanlab", {})
    if not options.get("enabled"):
        return None
    try:
        import swanlab
    except ImportError:
        print("提示：未安装 swanlab，跳过云端记录；pip install swanlab 后自动接入")
        return None
    try:
        handle = swanlab.init(
            project=options.get("project", "rsi-a-cpu"),
            experiment_name=options.get("experiment", "harness"),
            mode=options.get("mode", "local"),
            config={"task": "A-gas-drift", "baseline_f1": 0.7140297220788709},
        )
        print("SwanLab 已连接：project=%s，mode=%s" % (
            options.get("project"), options.get("mode", "local")))
        return handle
    except Exception as error:
        print("SwanLab 连接失败（不影响本地闭环）：%s" % error)
        return None


def log_round(handle, record):
    if handle is None:
        return
    try:
        payload = {
            "round": record["round"],
            "macro_f1": float(record["macro_f1_after"]),
            "seconds": record["wall_seconds"],
            "deployed": 1 if record.get("deployed") else 0,
            "prompt_tokens": record.get("prompt_tokens", 0),
            "cost_cny": record.get("cost_cny", 0.0),
        }
        handle.log(payload)
    except Exception as error:
        print("SwanLab 记录失败（不影响本地结果）：%s" % error)


def finish(handle, final):
    if handle is None:
        return
    try:
        handle.log({"final_macro_f1": final["final_macro_f1"],
                    "meets_baseline": 1 if final["meets_static_baseline"] else 0,
                    "total_wall_seconds": final["total_wall_seconds"]})
        handle.finish()
        print("SwanLab 记录已结束")
    except Exception as error:
        print("SwanLab 收尾失败（不影响本地结果）：%s" % error)
