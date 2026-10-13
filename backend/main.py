# -*- coding: utf-8 -*-
"""命令行入口：`python -m backend.main algorithms` / `demo`。

原先的 demo 依赖 scikit-learn 提供的 iris/diabetes 内置样例，没装 sklearn
就整段崩掉。现在平台登记了 8 个真实数据集，demo 改走它们，不再需要 sklearn。
"""
from __future__ import annotations

import argparse

import numpy as np

from backend.algorithms.registry import list_algorithms
from backend.data.registry import DatasetUnavailable, dataset_names
from backend.services.experiment import ExperimentError, run_dataset_experiment
from backend.services.train_service import run_experiment


def _synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """不依赖任何文件与第三方库的合成数据，用来快速验证算法层。"""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(180, 4))
    y_class = np.where(X[:, 0] + X[:, 1] > 0, "positive", "negative")
    y_reg = 2.0 * X[:, 0] - 0.7 * X[:, 1] + rng.normal(scale=0.1, size=len(X))
    return X[:140], X[140:], y_class, y_reg


def run_demo() -> None:
    X_train, X_test, y_class, y_reg = _synthetic_data()

    print("已注册算法：")
    for item in list_algorithms():
        print(f"- {item['name']}: {item['display_name']} ({item['task_type']})")

    print("\n合成数据上的单次实验：")
    for name, params, y in [
        ("gaussian_naive_bayes", {}, y_class),
        ("random_forest", {"n_estimators": 12, "max_depth": 5}, y_class),
        ("gradient_boosting", {"task": "classification", "n_estimators": 15}, y_class),
        ("gradient_boosting", {"task": "regression", "n_estimators": 15}, y_reg),
        ("linear_regression", {}, y_reg),
        ("knn", {"k": 5}, y_class),
        ("decision_tree", {"max_depth": 4}, y_class),
        ("svm", {}, y_class),
    ]:
        result = run_experiment(name, X_train, X_test, y[:140], y[140:], **params)
        metrics = result.get("metrics", {})
        summary = {
            key: round(value, 4)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        print(
            f"  {name:<20} {result['task_type']:<15} "
            f"train={result['train_time_seconds']:.4f}s "
            f"predict={result['predict_time_seconds']:.4f}s {summary}"
        )

    names = dataset_names()
    print(f"\n已登记数据集（{len(names)} 个）：{', '.join(names)}")
    print("\n真实数据集上的完整链路（加载 → 填充 → 标准化 → 训练 → 指标）：")
    plan = [
        ("iris", "knn", {"k": 5}),
        ("car", "decision_tree", {"max_depth": 6}),
        ("credit", "random_forest", {"n_estimators": 20, "max_depth": 6}),
        ("ckd", "gaussian_naive_bayes", {}),
        ("bike_day", "linear_regression", {}),
        ("concrete", "random_forest", {"n_estimators": 20, "task": "regression"}),
        ("iris", "kmeans", {"n_clusters": 3}),
    ]
    for dataset, algorithm, params in plan:
        try:
            result = run_dataset_experiment(dataset, algorithm, params, max_predictions=5)
        except DatasetUnavailable as exc:
            print(f"  跳过 {dataset}：{exc}")
            continue
        except ExperimentError as exc:
            print(f"  跳过 {dataset}/{algorithm}：{exc}")
            continue
        metrics = result.get("metrics") or {}
        summary = {
            key: round(value, 4)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        print(
            f"  {dataset:<11} {algorithm:<20} {result['task_type']:<15} "
            f"train={result['train_time_seconds']:.4f}s "
            f"predict={result['predict_time_seconds']:.4f}s {summary}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="ML visualization platform backend")
    parser.add_argument("command", choices=["demo", "algorithms"], nargs="?", default="demo")
    args = parser.parse_args()
    if args.command == "algorithms":
        for item in list_algorithms():
            print(item)
    else:
        run_demo()


if __name__ == "__main__":
    main()
