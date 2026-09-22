from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base_model import BaseModel, as_float_matrix, require_y


@dataclass
class _Node:
    value: float
    feature: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None


class _DecisionTree:
    def __init__(self, max_depth: int | None, min_samples_split: int, max_features: int, rng: np.random.Generator, task: str):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng
        self.task = task

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_DecisionTree":
        self.root = self._grow(X, y, depth=0)
        return self

    def _leaf_value(self, y: np.ndarray) -> float:
        if self.task == "regression":
            return float(np.mean(y))
        counts = np.bincount(y.astype(int), minlength=self.n_classes_)
        return float(np.argmax(counts))

    def _impurity(self, y: np.ndarray) -> float:
        if self.task == "regression":
            return float(np.sum((y - y.mean()) ** 2))
        counts = np.bincount(y.astype(int), minlength=self.n_classes_)
        probabilities = counts / len(y)
        return float(len(y) * (1.0 - np.sum(probabilities**2)))

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None]:
        features = self.rng.choice(X.shape[1], size=min(self.max_features, X.shape[1]), replace=False)
        parent_impurity = self._impurity(y)
        best_gain, best_feature, best_threshold = 0.0, None, None
        for feature in features:
            values = X[:, feature]
            order = np.argsort(values)
            sorted_values, sorted_y = values[order], y[order]
            boundaries = np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1
            for split_at in boundaries:
                left_y, right_y = sorted_y[:split_at], sorted_y[split_at:]
                gain = parent_impurity - self._impurity(left_y) - self._impurity(right_y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature)
                    best_threshold = float((sorted_values[split_at - 1] + sorted_values[split_at]) / 2.0)
        return best_feature, best_threshold

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        node = _Node(value=self._leaf_value(y))
        stop = len(y) < self.min_samples_split or len(np.unique(y)) <= 1
        if self.max_depth is not None and depth >= self.max_depth:
            stop = True
        if stop:
            return node
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return node
        left_mask = X[:, feature] <= threshold
        node.feature, node.threshold = feature, threshold
        node.left = self._grow(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _predict_one(self, row: np.ndarray, node: _Node) -> float:
        if node.feature is None:
            return node.value
        child = node.left if row[node.feature] <= node.threshold else node.right
        return self._predict_one(row, child)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray([self._predict_one(row, self.root) for row in X])


class RandomForest(BaseModel):
    """Random forest for classification or regression using bootstrap trees."""

    algorithm_name = "random_forest"

    def __init__(self, n_estimators: int = 50, max_depth: int | None = 8, min_samples_split: int = 2, max_features: int | str | None = "sqrt", task: str = "classification", random_state: int | None = 42) -> None:
        if n_estimators < 1 or min_samples_split < 2:
            raise ValueError("n_estimators must be positive and min_samples_split must be at least 2")
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.task = task
        self.task_type = task
        self.random_state = random_state

    def _resolve_max_features(self, n_features: int) -> int:
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, int):
            return max(1, min(self.max_features, n_features))
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError("max_features must be None, an integer, 'sqrt', or 'log2'")

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "RandomForest":
        X = as_float_matrix(X)
        y = require_y(y, len(X))
        rng = np.random.default_rng(self.random_state)
        if self.task == "classification":
            self.classes_, encoded = np.unique(y, return_inverse=True)
            train_y = encoded
        else:
            train_y = y.astype(float)
        max_features = self._resolve_max_features(X.shape[1])
        self.trees_ = []
        for _ in range(self.n_estimators):
            indices = rng.integers(0, len(X), size=len(X))
            tree_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            tree = _DecisionTree(self.max_depth, self.min_samples_split, max_features, tree_rng, self.task)
            if self.task == "classification":
                tree.n_classes_ = len(self.classes_)
            tree.fit(X[indices], train_y[indices])
            self.trees_.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "trees_"):
            raise RuntimeError("fit must be called before predict")
        X = as_float_matrix(X)
        predictions = np.vstack([tree.predict(X) for tree in self.trees_])
        if self.task == "regression":
            return predictions.mean(axis=0)
        encoded = np.asarray([
            np.bincount(row.astype(int), minlength=len(self.classes_)).argmax()
            for row in predictions.T
        ])
        return self.classes_[encoded]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise AttributeError("predict_proba is only available for classification")
        X = as_float_matrix(X)
        predictions = np.vstack([tree.predict(X).astype(int) for tree in self.trees_])
        return np.column_stack([np.mean(predictions == class_index, axis=0) for class_index in range(len(self.classes_))])
