from __future__ import annotations

import numpy as np

from .base_model import BaseModel, as_float_matrix, require_y


class GaussianNaiveBayes(BaseModel):
    """Gaussian Naive Bayes classifier implemented with log probabilities."""

    algorithm_name = "gaussian_naive_bayes"
    task_type = "classification"

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        if var_smoothing <= 0:
            raise ValueError("var_smoothing must be positive")
        self.var_smoothing = var_smoothing

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "GaussianNaiveBayes":
        X = as_float_matrix(X)
        y = require_y(y, len(X))
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.class_count_ = np.bincount(encoded).astype(float)
        self.class_prior_ = self.class_count_ / len(y)
        self.theta_ = np.vstack([X[encoded == i].mean(axis=0) for i in range(len(self.classes_))])
        self.var_ = np.vstack([X[encoded == i].var(axis=0) for i in range(len(self.classes_))])
        self.var_ = np.maximum(self.var_, self.var_smoothing * max(np.var(X, axis=0).mean(), np.finfo(float).eps))
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "classes_"):
            raise RuntimeError("fit must be called before prediction")
        X = as_float_matrix(X)
        log_density = -0.5 * (np.log(2.0 * np.pi * self.var_)[None, :, :] + ((X[:, None, :] - self.theta_[None, :, :]) ** 2) / self.var_[None, :, :])
        return log_density.sum(axis=2) + np.log(self.class_prior_)[None, :]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_likelihood = self._joint_log_likelihood(X)
        shifted = log_likelihood - log_likelihood.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]
