from __future__ import annotations

import numpy as np


def train_test_split(X: np.ndarray, y: np.ndarray | None = None, test_size: float = 0.2, random_state: int = 42) -> tuple[np.ndarray, ...]:
    """Deterministically split aligned arrays without leaking test data into training."""
    X = np.asarray(X)
    if X.ndim != 2 or len(X) == 0:
        raise ValueError("X must be a non-empty 2-dimensional array")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if y is not None and len(y) != len(X):
        raise ValueError("X and y must contain the same number of samples")
    indices = np.random.default_rng(random_state).permutation(len(X))
    test_count = max(1, int(round(len(X) * test_size)))
    test_indices, train_indices = indices[:test_count], indices[test_count:]
    if y is None:
        return X[train_indices], X[test_indices]
    y = np.asarray(y)
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def standardize(X_train: np.ndarray, X_test: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
    """Fit scaling statistics on training data only and apply them to both splits."""
    X_train = np.asarray(X_train, dtype=float)
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    transformed_train = (X_train - mean) / scale
    if X_test is None:
        return transformed_train, mean, scale
    return transformed_train, (np.asarray(X_test, dtype=float) - mean) / scale, mean, scale
