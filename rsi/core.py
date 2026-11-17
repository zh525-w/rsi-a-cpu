"""通用小模型：路径化特征抽取 + MLP 分类/回归。

设计要点：
- 特征按“路径”描述（[["key","features"],["index",0]]），因此任意嵌套的 inputs 都能吃，
  A / B / C 三类数据集用同一套代码，不需要按数据集写死字段名。
- 检查点保持 w1/b1/w2/b2/mean/scale/adam_* 的命名，与官方 baseline（contest.cpu-mlp.v1）
  完全兼容：A 数据集上可以直接继承官方父模型并续训。
- 分类走 tanh 隐层 + softmax，回归走 tanh 隐层 + softplus（保证输出非负）。
"""
import json
import math

import numpy as np

FORMAT = "contest.cpu-mlp.v1"
DEFAULT_HIDDEN = 32
MAX_FEATURES = 4096


# ---------------------------------------------------------------- 特征路径
def _walk(node, prefix, out):
    if isinstance(node, dict):
        for key in sorted(node, key=lambda item: str(item)):
            _walk(node[key], prefix + [["key", str(key)]], out)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _walk(value, prefix + [["index", index]], out)
    elif isinstance(node, bool):
        out.append(prefix)
    elif isinstance(node, (int, float)):
        out.append(prefix)


def leaf_paths(inputs, limit=MAX_FEATURES):
    """列出一份 inputs 里所有数值叶子的路径，顺序稳定。"""
    found = []
    _walk(inputs, [], found)
    return found[:limit]


def union_paths(rows, limit=MAX_FEATURES):
    """多行样本的路径并集，避免个别行字段缺失导致越界。"""
    merged = {}
    for row in rows:
        for path in leaf_paths(row.get("inputs"), limit):
            merged.setdefault(json.dumps(path, sort_keys=True), path)
            if len(merged) >= limit:
                return [merged[key] for key in sorted(merged)]
    return [merged[key] for key in sorted(merged)]


def value_at(inputs, path):
    current = inputs
    for kind, key in path:
        if isinstance(current, dict):
            if kind != "key" or key not in current:
                return None
            current = current[key]
        elif isinstance(current, (list, tuple)):
            if kind != "index":
                return None
            index = int(key)
            if index < 0 or index >= len(current):
                return None
            current = current[index]
        else:
            return None
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    value = float(current)
    return value if math.isfinite(value) else None


def extract(inputs, paths, fill=0.0):
    vector = np.empty(len(paths), dtype=np.float64)
    for index, path in enumerate(paths):
        value = value_at(inputs, path)
        vector[index] = fill if value is None else value
    return vector


def matrix(rows, paths, fill=0.0):
    return np.vstack([extract(row.get("inputs"), paths, fill) for row in rows]) if rows else np.zeros((0, len(paths)))


# ---------------------------------------------------------------- 模型
def _softplus(z):
    return np.logaddexp(0.0, z)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


class Net(object):
    """一个小 MLP；kind 取 classification 或 regression。"""

    def __init__(self, n_features, hidden=DEFAULT_HIDDEN, out_dim=6, kind="classification", seed=0):
        rng = np.random.default_rng(seed)
        self.kind = kind
        self.hidden = int(hidden)
        self.out_dim = int(out_dim)
        self.w1 = rng.normal(0.0, 0.1, (n_features, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.w2 = rng.normal(0.0, 0.1, (self.hidden, self.out_dim))
        self.b2 = np.zeros(self.out_dim)
        self.adam_first_w1 = np.zeros_like(self.w1)
        self.adam_first_b1 = np.zeros_like(self.b1)
        self.adam_first_w2 = np.zeros_like(self.w2)
        self.adam_first_b2 = np.zeros_like(self.b2)
        self.adam_second_w1 = np.zeros_like(self.w1)
        self.adam_second_b1 = np.zeros_like(self.b1)
        self.adam_second_w2 = np.zeros_like(self.w2)
        self.adam_second_b2 = np.zeros_like(self.b2)
        self.mean = np.zeros(n_features)
        self.scale = np.ones(n_features)
        self.classes = []
        self.paths = []
        self.step = 0

    # 前向 -----------------------------------------------------------
    def _hidden(self, x):
        return np.tanh(x.dot(self.w1) + self.b1)

    def logits(self, x):
        return self._hidden(x).dot(self.w2) + self.b2

    def predict(self, x):
        z = self.logits(x)
        if self.kind == "regression":
            return _softplus(z)[:, 0]
        shifted = z - z.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    # 反传 -----------------------------------------------------------
    def _backward(self, x, grad_z):
        h = self._hidden(x)
        grad_w2 = h.T.dot(grad_z)
        grad_b2 = grad_z.sum(axis=0)
        grad_h = grad_z.dot(self.w2.T) * (1.0 - h * h)
        grad_w1 = x.T.dot(grad_h)
        grad_b1 = grad_h.sum(axis=0)
        return grad_w1, grad_b1, grad_w2, grad_b2

    def fit(self, x, y, epochs, learning_rate, batch_size=128, seed=0, clip=5.0, min_delta=0.0):
        """Adam 训练；y 是分类的整数下标或回归的目标值。"""
        rng = np.random.default_rng(seed)
        total = x.shape[0]
        for epoch in range(int(epochs)):
            order = rng.permutation(total)
            for start in range(0, total, batch_size):
                index = order[start:start + batch_size]
                xb, yb = x[index], y[index]
                z = self.logits(xb)
                if self.kind == "regression":
                    pred = _softplus(z)
                    grad_z = (pred - yb[:, None]) * _sigmoid(z) / max(len(yb), 1)
                else:
                    shifted = z - z.max(axis=1, keepdims=True)
                    exp = np.exp(shifted)
                    prob = exp / exp.sum(axis=1, keepdims=True)
                    prob[np.arange(len(yb)), yb] -= 1.0
                    grad_z = prob / max(len(yb), 1)
                grads = self._backward(xb, grad_z)
                self._adam(grads, learning_rate, clip)
        return self

    def _adam(self, grads, learning_rate, clip, beta1=0.9, beta2=0.999, eps=1e-8):
        self.step += 1
        names = ("w1", "b1", "w2", "b2")
        for name, grad in zip(names, grads):
            grad = np.clip(grad, -clip, clip)
            first = getattr(self, "adam_first_" + name)
            second = getattr(self, "adam_second_" + name)
            first *= beta1
            first += (1.0 - beta1) * grad
            second *= beta2
            second += (1.0 - beta2) * grad * grad
            corrected_first = first / (1.0 - beta1 ** self.step)
            corrected_second = second / (1.0 - beta2 ** self.step)
            param = getattr(self, name)
            param -= learning_rate * corrected_first / (np.sqrt(corrected_second) + eps)

    # 存档 -----------------------------------------------------------
    def save(self, path, extra=None):
        meta = {
            "format": FORMAT,
            "kind": self.kind,
            "hidden": self.hidden,
            "classes": list(self.classes),
            "paths": self.paths,
            "parameter_count": int(self.w1.size + self.b1.size + self.w2.size + self.b2.size),
            "feature_count": int(self.w1.shape[0]),
            "adam_steps": self.step,
        }
        if extra:
            meta.update(extra)
        np.savez_compressed(
            path,
            w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
            mean=self.mean, scale=self.scale,
            adam_first_w1=self.adam_first_w1, adam_first_b1=self.adam_first_b1,
            adam_first_w2=self.adam_first_w2, adam_first_b2=self.adam_first_b2,
            adam_second_w1=self.adam_second_w1, adam_second_b1=self.adam_second_b1,
            adam_second_w2=self.adam_second_w2, adam_second_b2=self.adam_second_b2,
            metadata=np.array(json.dumps(meta, ensure_ascii=False, sort_keys=True)),
        )

    @classmethod
    def load(cls, path):
        with np.load(str(path), allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            net = cls(
                data["w1"].shape[0],
                hidden=meta.get("hidden", data["w1"].shape[1]),
                out_dim=data["w2"].shape[1],
                kind=meta.get("kind", "classification"),
            )
            for name in ("w1", "b1", "w2", "b2", "mean", "scale",
                         "adam_first_w1", "adam_first_b1", "adam_first_w2", "adam_first_b2",
                         "adam_second_w1", "adam_second_b1", "adam_second_w2", "adam_second_b2"):
                if name in data:
                    setattr(net, name, np.array(data[name], dtype=np.float64))
            net.classes = list(meta.get("classes") or [])
            net.paths = list(meta.get("paths") or [])
            net.step = int(meta.get("adam_steps", 0))
            return net, meta


# ---------------------------------------------------------------- 标准化与指标
def fit_scaler(x):
    mean = x.mean(axis=0) if x.size else np.zeros(x.shape[1])
    scale = x.std(axis=0) if x.size else np.ones(x.shape[1])
    scale = np.where(scale < 1e-6, 1.0, scale)
    return mean, scale


def macro_f1(labels, predicted, classes):
    labels = np.asarray(labels)
    predicted = np.asarray(predicted)
    scores = []
    for index in range(len(classes)):
        true_positive = int(((labels == index) & (predicted == index)).sum())
        false_positive = int(((labels != index) & (predicted == index)).sum())
        false_negative = int(((labels == index) & (predicted != index)).sum())
        if true_positive == 0:
            scores.append(0.0)
            continue
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        scores.append(2 * precision * recall / (precision + recall))
    return float(sum(scores) / len(scores)) if scores else 0.0


def accuracy(labels, predicted):
    if len(labels) == 0:
        return 0.0
    return float((np.asarray(labels) == np.asarray(predicted)).mean())


def mean_absolute_error(target, predicted):
    if len(target) == 0:
        return 0.0
    return float(np.abs(np.asarray(target) - np.asarray(predicted)).mean())


def root_mean_squared_error(target, predicted):
    if len(target) == 0:
        return 0.0
    return float(np.sqrt(((np.asarray(target) - np.asarray(predicted)) ** 2).mean()))


def score_rows(net, x, answers, classes, task):
    pred = net.predict(x)
    if task == "regression":
        return root_mean_squared_error(answers, pred)
    return macro_f1(answers, pred.argmax(axis=1), classes)
