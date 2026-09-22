from __future__ import annotations

import numpy as np

from .base_model import BaseModel, as_float_matrix, require_y


class _Node:
    """Internal node stores a split rule; leaf nodes store class counts."""

    __slots__ = ("feature", "threshold", "left", "right", "counts")

    def __init__(self, feature=None, threshold=None, left=None, right=None, counts=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.counts = counts


def _gini(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    return 1.0 - float(np.sum(probs**2))


class DecisionTree(BaseModel):
    """CART decision tree classifier using the Gini index as split criterion.

    At every node the tree scans all features and candidate thresholds
    (midpoints between consecutive unique values) and keeps the split
    with the lowest weighted Gini. max_depth / min_samples_split guard
    against overfitting. predict_proba returns leaf class frequencies.
    """

    algorithm_name = "decision_tree"
    task_type = "classification"

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "DecisionTree":
        X = as_float_matrix(X)
        y = require_y(y, len(X))
        self.classes_, encoded = np.unique(y, return_inverse=True)
        one_hot = np.zeros((len(y), len(self.classes_)))
        one_hot[np.arange(len(y)), encoded] = 1.0
        self.root_ = self._build(X, one_hot, depth=0)
        return self

    def _best_split(self, X: np.ndarray, counts: np.ndarray) -> tuple[int | None, float | None]:
        n_samples = counts.sum()
        parent = _gini(counts)
        best_gini, best_feature, best_threshold = parent, None, None
        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            for threshold in (values[:-1] + values[1:]) / 2:
                mask = X[:, feature] <= threshold
                left_counts, right_counts = counts[mask].sum(axis=0), counts[~mask].sum(axis=0)
                if left_counts.sum() == 0 or right_counts.sum() == 0:
                    continue
                weighted = (left_counts.sum() * _gini(left_counts) + right_counts.sum() * _gini(right_counts)) / n_samples
                if weighted < best_gini:
                    best_gini, best_feature, best_threshold = weighted, feature, threshold
        return best_feature, best_threshold

    def _build(self, X: np.ndarray, counts: np.ndarray, depth: int) -> _Node:
        if depth >= self.max_depth or counts.sum() < self.min_samples_split or _gini(counts.sum(axis=0)) == 0.0:
            return _Node(counts=counts.sum(axis=0))
        feature, threshold = self._best_split(X, counts)
        if feature is None:
            return _Node(counts=counts.sum(axis=0))
        mask = X[:, feature] <= threshold
        left = self._build(X[mask], counts[mask], depth + 1)
        right = self._build(X[~mask], counts[~mask], depth + 1)
        return _Node(feature=feature, threshold=threshold, left=left, right=right)

    def _leaf_counts(self, x: np.ndarray, node: _Node) -> np.ndarray:
        if node.counts is not None:
            return node.counts
        if x[node.feature] <= node.threshold:
            return self._leaf_counts(x, node.left)
        return self._leaf_counts(x, node.right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "root_"):
            raise RuntimeError("fit must be called before prediction")
        X = as_float_matrix(X)
        proba = np.vstack([self._leaf_counts(x, self.root_) for x in X])
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
