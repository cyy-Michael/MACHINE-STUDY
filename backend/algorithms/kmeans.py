from __future__ import annotations

import numpy as np

from .base_model import BaseModel, as_float_matrix


class KMeans(BaseModel):
    """K-Means clustering with k-means++ initialization."""

    algorithm_name = "kmeans"
    task_type = "clustering"

    def __init__(self, n_clusters: int = 3, max_iter: int = 300, tol: float = 1e-4, random_state: int | None = 42) -> None:
        if n_clusters < 1 or max_iter < 1 or tol < 0:
            raise ValueError("n_clusters and max_iter must be positive; tol cannot be negative")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _initialize_centers(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.n_clusters > len(X):
            raise ValueError("n_clusters cannot be greater than the number of samples")
        centers = [X[rng.integers(len(X))]]
        closest_dist_sq = np.sum((X - centers[0]) ** 2, axis=1)
        for _ in range(1, self.n_clusters):
            total = closest_dist_sq.sum()
            index = rng.integers(len(X)) if total == 0 else rng.choice(len(X), p=closest_dist_sq / total)
            centers.append(X[index])
            closest_dist_sq = np.minimum(closest_dist_sq, np.sum((X - X[index]) ** 2, axis=1))
        return np.asarray(centers, dtype=float)

    @staticmethod
    def _assign(X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distances = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels, distances[np.arange(len(X)), labels]

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "KMeans":
        del y
        X = as_float_matrix(X)
        rng = np.random.default_rng(self.random_state)
        centers = self._initialize_centers(X, rng)
        previous_inertia = np.inf
        for iteration in range(1, self.max_iter + 1):
            labels, distances = self._assign(X, centers)
            new_centers = centers.copy()
            for cluster in range(self.n_clusters):
                members = X[labels == cluster]
                new_centers[cluster] = members.mean(axis=0) if len(members) else X[np.argmax(distances)]
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            inertia = float(distances.sum())
            if shift <= self.tol or abs(previous_inertia - inertia) <= self.tol:
                break
            previous_inertia = inertia
        self.cluster_centers_ = centers
        self.labels_, final_distances = self._assign(X, centers)
        self.inertia_ = float(final_distances.sum())
        self.n_iter_ = iteration
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("fit must be called before predict")
        labels, _ = self._assign(as_float_matrix(X), self.cluster_centers_)
        return labels

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("fit must be called before transform")
        X = as_float_matrix(X)
        return np.sqrt(np.sum((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2, axis=2))

    def fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).labels_
