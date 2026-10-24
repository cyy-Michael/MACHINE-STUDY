# -*- coding: utf-8 -*-
"""指标计算：分类、回归、聚类。

聚类指标（轮廓系数、Calinski-Harabasz）是老师明确要求的，原实现完全没有。
轮廓系数是 O(n^2) 的，``day.csv`` 这种 700 行无所谓，``airquality`` 9000 行
就会把内存吃满，所以这里分块计算 + 超限抽样。
"""
from __future__ import annotations

import numpy as np

# 轮廓系数的样本上限：超过就按簇分层抽样，并在结果里标注被抽样过。
SILHOUETTE_MAX_SAMPLES = 5000


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | list[list[int]]]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    lookup = {label: index for index, label in enumerate(labels)}
    for actual, predicted in zip(y_true, y_pred):
        matrix[lookup[actual], lookup[predicted]] += 1
    accuracy = float(np.trace(matrix) / max(len(y_true), 1))
    precisions, recalls, f1s = [], [], []
    for index in range(len(labels)):
        true_positive = matrix[index, index]
        precision = true_positive / max(matrix[:, index].sum(), 1)
        recall = true_positive / max(matrix[index, :].sum(), 1)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(2 * precision * recall / max(precision + recall, np.finfo(float).eps))
    return {
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
        "confusion_matrix": matrix.tolist(),
        "labels": labels.tolist(),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    error = y_true - y_pred
    mse = float(np.mean(error**2))
    total = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(1.0 - np.sum(error**2) / max(total, np.finfo(float).eps)),
    }


# --------------------------------------------------------------------------
# 聚类指标
# --------------------------------------------------------------------------

def _subsample_by_cluster(
    labels: np.ndarray,
    limit: int,
    random_state: int,
) -> np.ndarray:
    """按簇分层抽样，保证每个簇都有代表。

    直接随机抽样可能整个丢掉一个小簇，指标就失去意义了。
    """
    unique = np.unique(labels)
    rng = np.random.default_rng(random_state)
    per_cluster = max(1, limit // len(unique))
    picked = []
    for label in unique:
        members = np.where(labels == label)[0]
        take = min(len(members), per_cluster)
        picked.append(rng.choice(members, size=take, replace=False))
    return np.concatenate(picked)


def silhouette_score(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    chunk_size: int = 256,
) -> float:
    """轮廓系数 s(i) = (b - a) / max(a, b) 的均值。

    用分块计算代替整张距离矩阵：n=9000 时整张矩阵是 648 MB，
    分块后每块只占 ``chunk_size * n`` 个浮点数。
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n = len(X)
    unique = np.unique(labels)
    n_clusters = len(unique)

    if n_clusters < 2:
        raise ValueError("轮廓系数需要至少 2 个簇")
    if n_clusters >= n:
        raise ValueError("每个样本各成一簇，轮廓系数无定义")

    squared = np.einsum("ij,ij->i", X, X)
    members = {label: np.where(labels == label)[0] for label in unique}
    code = np.searchsorted(unique, labels)

    scores = np.empty(n, dtype=float)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = X[start:stop]

        # ||a-b||^2 = |a|^2 + |b|^2 - 2·a·b，比显式循环快且不需要整张矩阵
        d2 = squared[start:stop, None] + squared[None, :] - 2.0 * (block @ X.T)
        np.maximum(d2, 0.0, out=d2)
        distances = np.sqrt(d2)

        # 到每个簇的平均距离
        means = np.empty((stop - start, n_clusters), dtype=float)
        for column, label in enumerate(unique):
            means[:, column] = distances[:, members[label]].mean(axis=1)

        rows = np.arange(stop - start)
        own = code[start:stop]
        a = means[rows, own]          # 花式索引返回副本，下面改写 means 不影响它
        means[rows, own] = np.inf
        b = means.min(axis=1)

        denominator = np.maximum(a, b)
        safe = np.where(denominator > 0, denominator, 1.0)
        scores[start:stop] = np.where(denominator > 0, (b - a) / safe, 0.0)

    return float(scores.mean())


def calinski_harabasz_score(X: np.ndarray, labels: np.ndarray) -> float:
    """CH 指数 = (簇间离散 / (k-1)) / (簇内离散 / (n-k))，越大越好。"""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n = len(X)
    unique = np.unique(labels)
    n_clusters = len(unique)

    if n_clusters < 2:
        raise ValueError("CH 指数需要至少 2 个簇")
    if n_clusters >= n:
        raise ValueError("每个样本各成一簇，CH 指数无定义")

    global_mean = X.mean(axis=0)
    between = 0.0
    within = 0.0
    for label in unique:
        cluster = X[labels == label]
        centroid = cluster.mean(axis=0)
        between += len(cluster) * float(np.sum((centroid - global_mean) ** 2))
        within += float(np.sum((cluster - centroid) ** 2))

    if within <= 0:
        # 簇内离散为 0 表示完美分离，数学上是无穷大
        return float("inf")
    return float((between / (n_clusters - 1)) / (within / (n - n_clusters)))


def clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    max_samples: int = SILHOUETTE_MAX_SAMPLES,
    random_state: int = 42,
) -> dict[str, object]:
    """聚类任务的全部指标。样本过多时轮廓系数按簇分层抽样。"""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique = np.unique(labels)

    result: dict[str, object] = {
        "n_clusters": int(len(unique)),
        "cluster_sizes": {str(label): int((labels == label).sum()) for label in unique},
    }

    if len(unique) < 2:
        result["silhouette"] = None
        result["calinski_harabasz"] = None
        result["note"] = "只有一个簇，轮廓系数与 CH 指数无定义"
        return result

    sampled = False
    if len(X) > max_samples:
        picked = _subsample_by_cluster(labels, max_samples, random_state)
        X_eval, labels_eval = X[picked], labels[picked]
        sampled = True
    else:
        X_eval, labels_eval = X, labels

    result["silhouette"] = silhouette_score(X_eval, labels_eval)
    result["calinski_harabasz"] = calinski_harabasz_score(X_eval, labels_eval)
    result["silhouette_sampled"] = sampled
    result["silhouette_sample_size"] = int(len(X_eval))
    return result
