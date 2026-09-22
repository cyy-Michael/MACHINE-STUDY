from __future__ import annotations

import numpy as np

from .base_model import BaseModel, as_float_matrix


class PCA(BaseModel):
    """Principal Component Analysis based on covariance eigendecomposition."""

    algorithm_name = "pca"
    task_type = "dimensionality_reduction"

    def __init__(self, n_components: int = 2) -> None:
        if n_components < 1:
            raise ValueError("n_components must be positive")
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "PCA":
        del y
        X = as_float_matrix(X)
        if self.n_components > min(X.shape):
            raise ValueError("n_components cannot exceed min(n_samples, n_features)")
        self.mean_ = X.mean(axis=0)
        centered = X - self.mean_
        covariance = centered.T @ centered / max(len(X) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        self.explained_variance_ = eigenvalues[order][: self.n_components]
        self.components_ = eigenvectors[:, order[: self.n_components]].T
        self.explained_variance_ratio_ = self.explained_variance_ / max(eigenvalues.sum(), np.finfo(float).eps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "components_"):
            raise RuntimeError("fit must be called before transform")
        X = as_float_matrix(X)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "components_"):
            raise RuntimeError("fit must be called before inverse_transform")
        X = as_float_matrix(X)
        if X.shape[1] != self.n_components:
            raise ValueError("X has a different number of components than this PCA model")
        return X @ self.components_ + self.mean_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)
