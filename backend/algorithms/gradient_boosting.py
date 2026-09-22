from __future__ import annotations

import numpy as np

from .base_model import BaseModel, as_float_matrix, require_y


class _RegressionTree:
    """Squared-error tree used as the weak learner for gradient boosting."""

    def __init__(self, max_depth: int, min_samples_split: int):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RegressionTree":
        self.root = self._grow(X, y, 0)
        return self

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        node = {"value": float(y.mean())}
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.ptp(y) == 0:
            return node
        parent_error = np.sum((y - y.mean()) ** 2)
        best_gain, best_feature, best_threshold = 0.0, None, None
        for feature in range(X.shape[1]):
            order = np.argsort(X[:, feature])
            values, sorted_y = X[order, feature], y[order]
            boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
            for index in boundaries:
                left, right = sorted_y[:index], sorted_y[index:]
                gain = parent_error - np.sum((left - left.mean()) ** 2) - np.sum((right - right.mean()) ** 2)
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, (values[index - 1] + values[index]) / 2.0
        if best_feature is None:
            return node
        mask = X[:, best_feature] <= best_threshold
        node.update({"feature": best_feature, "threshold": best_threshold})
        node["left"] = self._grow(X[mask], y[mask], depth + 1)
        node["right"] = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def _predict_one(self, row: np.ndarray, node: dict) -> float:
        if "feature" not in node:
            return node["value"]
        branch = "left" if row[node["feature"]] <= node["threshold"] else "right"
        return self._predict_one(row, node[branch])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray([self._predict_one(row, self.root) for row in X])


class GradientBoosting(BaseModel):
    """Gradient boosting for regression and binary/multiclass classification."""

    algorithm_name = "gradient_boosting"

    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.05, max_depth: int = 2, min_samples_split: int = 2, task: str = "regression", random_state: int | None = 42) -> None:
        if n_estimators < 1 or learning_rate <= 0 or max_depth < 1 or min_samples_split < 2:
            raise ValueError("invalid gradient boosting parameters")
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.task = task
        self.task_type = task
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "GradientBoosting":
        X = as_float_matrix(X)
        y = require_y(y, len(X))
        self.estimators_ = []
        rng = np.random.default_rng(self.random_state)
        if self.task == "regression":
            target = y.astype(float)
            self.init_ = float(target.mean())
            current = np.full(len(X), self.init_)
            for _ in range(self.n_estimators):
                tree = _RegressionTree(self.max_depth, self.min_samples_split).fit(X, target - current)
                current += self.learning_rate * tree.predict(X)
                self.estimators_.append(tree)
        else:
            self.classes_, encoded = np.unique(y, return_inverse=True)
            if len(self.classes_) < 2:
                raise ValueError("classification requires at least two classes")
            if len(self.classes_) == 2:
                rate = np.clip(np.mean(encoded), 1e-7, 1 - 1e-7)
                self.init_ = float(np.log(rate / (1 - rate)))
                current = np.full(len(X), self.init_)
                for _ in range(self.n_estimators):
                    probability = 1.0 / (1.0 + np.exp(-np.clip(current, -35, 35)))
                    tree = _RegressionTree(self.max_depth, self.min_samples_split).fit(X, encoded - probability)
                    current += self.learning_rate * tree.predict(X)
                    self.estimators_.append(tree)
            else:
                current = np.zeros((len(X), len(self.classes_)))
                for _ in range(self.n_estimators):
                    exp_scores = np.exp(current - current.max(axis=1, keepdims=True))
                    probability = exp_scores / exp_scores.sum(axis=1, keepdims=True)
                    stage = []
                    for class_index in range(len(self.classes_)):
                        residual = (encoded == class_index).astype(float) - probability[:, class_index]
                        tree = _RegressionTree(self.max_depth, self.min_samples_split).fit(X, residual)
                        current[:, class_index] += self.learning_rate * tree.predict(X)
                        stage.append(tree)
                    self.estimators_.append(stage)
        return self

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "estimators_"):
            raise RuntimeError("fit must be called before predict")
        X = as_float_matrix(X)
        if self.task == "regression":
            result = np.full(len(X), self.init_)
            for tree in self.estimators_:
                result += self.learning_rate * tree.predict(X)
            return result
        if len(self.classes_) == 2:
            result = np.full(len(X), self.init_)
            for tree in self.estimators_:
                result += self.learning_rate * tree.predict(X)
            return result
        result = np.zeros((len(X), len(self.classes_)))
        for stage in self.estimators_:
            for class_index, tree in enumerate(stage):
                result[:, class_index] += self.learning_rate * tree.predict(X)
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise AttributeError("predict_proba is only available for classification")
        raw = self._raw_predict(X)
        if len(self.classes_) == 2:
            positive = 1.0 / (1.0 + np.exp(-np.clip(raw, -35, 35)))
            return np.column_stack([1.0 - positive, positive])
        exp_scores = np.exp(raw - raw.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.task == "regression":
            return self._raw_predict(X)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
