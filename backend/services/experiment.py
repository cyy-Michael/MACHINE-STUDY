# -*- coding: utf-8 -*-
"""端到端实验编排。

把「选数据集 → 选算法 → 调参数 → 训练 → 看指标」这条链路串成一个函数，
产出符合 ``docs/BACKEND_PLAN.md`` 第七节契约的结果 JSON。

**处理顺序是按防泄露要求排的**：

1. 加载并编码（此时数值缺失仍是 ``np.nan``）
2. 划分训练/测试集
3. **用训练集均值填充缺失**，再套用到测试集
4. **用训练集统计量标准化**，再套用到测试集
5. 训练（计时）→ 预测（计时）→ 指标

第 3、4 步必须在划分之后，否则测试集信息会渗进训练，属于数据泄露。
"""
from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Callable

import numpy as np

from backend.algorithms.registry import ALGORITHM_REGISTRY
from backend.data.encoding import impute_with_train_statistics
from backend.data.preprocessing import standardize, train_test_split
from backend.data.registry import get_spec, load_dataset
from .train_service import run_experiment

# 结果 JSON 里最多返回多少条预测；csv 大的数据集动辄上万条，全塞进响应没意义
DEFAULT_MAX_PREDICTIONS = 200

# 二维散点最多画多少个点
MAX_SCATTER_POINTS = 500

# 进度回调签名：(消息, 0..1 的进度)
ProgressCallback = Callable[[str, float], None]


def _emit(progress: ProgressCallback | None, message: str, fraction: float) -> None:
    if progress is not None:
        progress(message, fraction)


def json_safe(value: Any) -> Any:
    """把结果递归转成严格合法的 JSON 值。

    ``json.dumps`` 默认会把 ``inf``/``nan`` 写成裸的 ``Infinity``/``NaN``，
    这不是合法 JSON —— 浏览器的 ``JSON.parse`` 会直接抛错。聚类完美分离时
    Calinski-Harabasz 正好返回 ``inf``，所以必须在出口处拦掉。

    numpy 标量也一并转成原生类型，否则 Starlette 的序列化器会报错。
    """
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


class ExperimentError(RuntimeError):
    """实验无法执行（数据集/算法不匹配、参数非法等）。"""


SUPERVISED_TASKS = ("classification", "regression")


def algorithm_task_type(name: str) -> str:
    if name not in ALGORITHM_REGISTRY:
        raise ExperimentError(
            f"未知算法 {name!r}；可选：{sorted(ALGORITHM_REGISTRY)}"
        )
    return str(ALGORITHM_REGISTRY[name]["task_type"])


def _validate_pair(dataset_task: str, algorithm: str) -> str:
    """校验算法与数据集是否匹配，并返回算法具体的任务类型。

    ``random_forest`` / ``gradient_boosting`` 的 task_type 是
    "classification / regression"，靠构造参数 ``task`` 切换，这里要把它
    收敛成具体任务，否则后面无法判断该用哪套指标。

    聚类与降维**不依赖标签**，所以任何数据集都能跑——这不是不匹配。
    """
    declared = algorithm_task_type(algorithm)

    if declared in ("clustering", "dimensionality_reduction"):
        return declared

    if declared == "classification / regression":
        if dataset_task not in SUPERVISED_TASKS:
            raise ExperimentError(
                f"算法 {algorithm!r} 只支持分类或回归，数据集任务是 {dataset_task!r}"
            )
        return dataset_task

    if dataset_task in SUPERVISED_TASKS and declared != dataset_task:
        raise ExperimentError(
            f"算法 {algorithm!r} 用于 {declared}，与数据集的 {dataset_task} 任务不匹配"
        )
    return declared


def _project_2d(X: np.ndarray) -> np.ndarray:
    """把特征投影到 2 维用于散点图；多于 2 维时取前两个主成分。

    直接拿前两个原始特征画图会掩盖真实结构，PCA 投影更能反映聚类形态。
    """
    X = np.asarray(X, dtype=float)
    if X.shape[1] == 2:
        return X
    if X.shape[1] == 1:
        return np.column_stack([X[:, 0], np.zeros(len(X))])
    centered = X - X.mean(axis=0)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    return centered @ components[:2].T


def _build_visualization(
    task_type: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    """按任务类型给出前端画图需要的数据。"""
    if task_type == "regression":
        pairs = np.column_stack([np.asarray(y_test, dtype=float), np.asarray(predictions, dtype=float)])
        if len(pairs) > MAX_SCATTER_POINTS:
            step = max(1, len(pairs) // MAX_SCATTER_POINTS)
            pairs = pairs[::step]
        return {"true_vs_pred": pairs.tolist()}

    projected = _project_2d(np.asarray(X_test, dtype=float))
    labels = np.asarray(predictions, dtype=float)
    points = np.column_stack([projected, labels])
    if len(points) > MAX_SCATTER_POINTS:
        step = max(1, len(points) // MAX_SCATTER_POINTS)
        points = points[::step]
    return {"scatter_2d": points.tolist(), "projection": "pca" if X_test.shape[1] > 2 else "raw"}


def make_experiment_id(dataset: str, algorithm: str) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{algorithm}-{dataset}"


def run_dataset_experiment(
    dataset: str,
    algorithm: str,
    parameters: dict[str, Any] | None = None,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    standardize_features: bool = True,
    max_predictions: int = DEFAULT_MAX_PREDICTIONS,
    experiment_id: str | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """跑一次完整实验并返回结果 JSON。

    出错时抛 :class:`ExperimentError`（数据/算法不匹配）或数据集自身的异常，
    由调用方决定是变成 400 还是 500。
    """
    if not 0 < test_size < 1:
        raise ExperimentError(f"test_size 必须在 0 和 1 之间，收到 {test_size}")

    parameters = dict(parameters or {})

    for key in ("task",):
        if key in parameters and algorithm not in ("random_forest", "gradient_boosting"):
            raise ExperimentError(f"算法 {algorithm!r} 不接受参数 {key!r}")

    # 先用元数据校验（不碰文件）：参数或算法不匹配时应当立刻返回，
    # 不该为了报一个 400 先去读一遍几 MB 的数据文件。
    spec = get_spec(dataset)
    task_type = _validate_pair(spec.task_type, algorithm)

    _emit(progress, f"加载数据集 {dataset}", 0.05)
    loaded = load_dataset(dataset)
    _emit(
        progress,
        f"读取 {loaded.table.format} 文件完成：{len(loaded.table.rows)} 行，"
        f"编码后 {loaded.X.shape[1]} 个特征",
        0.15,
    )

    # 1) 划分——必须先于填充和标准化
    _emit(progress, "划分训练集与测试集", 0.25)
    split = train_test_split(
        loaded.X, loaded.y, test_size=test_size, random_state=random_state
    )
    X_train, X_test, y_train, y_test = split

    # 2) 缺失值填充：统计量只来自训练集
    _emit(progress, "填充缺失值（只用训练集均值）", 0.35)
    X_train, X_test, _ = impute_with_train_statistics(X_train, X_test)

    # 3) 标准化：同样只来自训练集
    scaler = None
    if standardize_features:
        _emit(progress, "标准化特征（只用训练集统计量）", 0.45)
        X_train, X_test, scaler_mean, scaler_scale = standardize(X_train, X_test)
        scaler = {"mean": scaler_mean.tolist(), "scale": scaler_scale.tolist()}

    # 4) 训练与预测（内部已分开计时）
    is_clustering = task_type in ("clustering", "dimensionality_reduction")
    _emit(progress, f"训练 {algorithm}", 0.55)
    raw = run_experiment(
        algorithm,
        X_train,
        X_test,
        None if is_clustering else y_train,
        None if is_clustering else y_test,
        **parameters,
    )
    _emit(progress, "计算指标", 0.85)

    predictions = np.asarray(raw["predictions"])
    result: dict[str, Any] = {
        "experiment_id": experiment_id or make_experiment_id(dataset, algorithm),
        "status": "finished",
        "dataset": dataset,
        "algorithm": algorithm,
        "task_type": raw["task_type"],
        "parameters": raw["parameters"],
        "split": {
            "test_size": test_size,
            "random_state": random_state,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_features": int(loaded.X.shape[1]),
        },
        "preprocessing": {
            "imputed": True,
            "impute_strategy": "train_mean",
            "standardized": bool(standardize_features),
            "encoded_missing": int(np.isnan(loaded.X).sum()),
            "dropped_rows": loaded.dropped_rows,
            "scaler": scaler,
        },
        "metrics": raw.get("metrics"),
        "train_time_seconds": raw["train_time_seconds"],
        "predict_time_seconds": raw["predict_time_seconds"],
        "predictions": predictions[:max_predictions].tolist(),
        "predictions_total": int(len(predictions)),
        "predictions_truncated": bool(len(predictions) > max_predictions),
        "visualization": _build_visualization(task_type, X_test, y_test, predictions),
        "error": None,
    }

    if task_type in SUPERVISED_TASKS:
        result["label_names"] = loaded.label_names
        if task_type == "classification" and result["metrics"]:
            # 把混淆矩阵的整数下标映射回可读标签名，前端直接显示
            metrics = dict(result["metrics"])
            metrics["label_names"] = loaded.label_names
            result["metrics"] = metrics
        result["y_test_sample"] = np.asarray(y_test)[:max_predictions].tolist()

    _emit(progress, "实验完成", 1.0)
    return json_safe(result)
