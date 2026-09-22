from __future__ import annotations

import argparse

import numpy as np

from backend.algorithms.registry import list_algorithms
from backend.services.train_service import run_experiment


def _demo_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(180, 4))
    y_class = np.where(X[:, 0] + X[:, 1] > 0, "positive", "negative")
    y_reg = 2.0 * X[:, 0] - 0.7 * X[:, 1] + rng.normal(scale=0.1, size=len(X))
    return X[:140], X[140:], y_class, y_reg


def run_demo() -> None:
    X_train, X_test, y_class, y_reg = _demo_data()
    print("Registered algorithms:")
    for item in list_algorithms():
        print(f"- {item['name']}: {item['display_name']}")
    for name, params, y in [
        ("gaussian_naive_bayes", {}, y_class),
        ("random_forest", {"n_estimators": 12, "max_depth": 5}, y_class),
        ("gradient_boosting", {"task": "classification", "n_estimators": 15}, y_class),
        ("gradient_boosting", {"task": "regression", "n_estimators": 15}, y_reg),
    ]:
        result = run_experiment(name, X_train, X_test, y[:140], y[140:], **params)
        print(f"{name} ({result['task_type']}): {result.get('metrics', {})}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ML visualization platform backend bootstrap")
    parser.add_argument("command", choices=["demo", "algorithms"], nargs="?", default="demo")
    args = parser.parse_args()
    if args.command == "algorithms":
        for item in list_algorithms():
            print(item)
    else:
        run_demo()


if __name__ == "__main__":
    main()
