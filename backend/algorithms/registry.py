from __future__ import annotations

from typing import Any, Callable

from .gradient_boosting import GradientBoosting
from .kmeans import KMeans
from .naive_bayes import GaussianNaiveBayes
from .pca import PCA
from .random_forest import RandomForest


AlgorithmFactory = Callable[..., Any]

ALGORITHM_REGISTRY: dict[str, dict[str, Any]] = {
    "kmeans": {"display_name": "K-平均聚类", "task_type": "clustering", "implemented_in_house": True, "factory": KMeans, "default_params": {"n_clusters": 3, "max_iter": 300, "tol": 1e-4, "random_state": 42}},
    "random_forest": {"display_name": "随机森林", "task_type": "classification / regression", "implemented_in_house": True, "factory": RandomForest, "default_params": {"n_estimators": 50, "max_depth": 8, "max_features": "sqrt", "task": "classification"}},
    "gaussian_naive_bayes": {"display_name": "高斯朴素贝叶斯", "task_type": "classification", "implemented_in_house": True, "factory": GaussianNaiveBayes, "default_params": {"var_smoothing": 1e-9}},
    "pca": {"display_name": "主成分分析（降维）", "task_type": "dimensionality_reduction", "implemented_in_house": True, "factory": PCA, "default_params": {"n_components": 2}},
    "gradient_boosting": {"display_name": "梯度增强", "task_type": "classification / regression", "implemented_in_house": True, "factory": GradientBoosting, "default_params": {"n_estimators": 50, "learning_rate": 0.05, "max_depth": 2, "task": "regression"}},
}


def list_algorithms(task_type: str | None = None) -> list[dict[str, Any]]:
    result = []
    for key, info in ALGORITHM_REGISTRY.items():
        if task_type and task_type not in info["task_type"]:
            continue
        result.append({"name": key, **{k: v for k, v in info.items() if k != "factory"}})
    return result


def create_algorithm(name: str, **params: Any) -> Any:
    try:
        factory = ALGORITHM_REGISTRY[name]["factory"]
    except KeyError as exc:
        raise ValueError(f"unknown algorithm: {name}") from exc
    return factory(**params)
