from __future__ import annotations

import numpy as np


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
