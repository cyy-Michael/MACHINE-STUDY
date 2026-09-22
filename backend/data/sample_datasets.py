from __future__ import annotations

import numpy as np


def _load_watermelon() -> tuple[np.ndarray, np.ndarray, str, list[str]]:
    """Watermelon dataset 3.0alpha (Zhou, Machine Learning): 17 samples,
    2 continuous features (density, sugar ratio), binary label. The data
    ships in code so demos run without any download."""
    raw = [
        [0.697, 0.460, 1], [0.774, 0.376, 1], [0.634, 0.264, 1],
        [0.608, 0.318, 1], [0.556, 0.215, 1], [0.403, 0.237, 1],
        [0.481, 0.149, 1], [0.437, 0.211, 1], [0.666, 0.091, 0],
        [0.243, 0.267, 0], [0.245, 0.057, 0], [0.343, 0.099, 0],
        [0.639, 0.161, 0], [0.657, 0.198, 0], [0.360, 0.370, 0],
        [0.593, 0.042, 0], [0.719, 0.103, 0],
    ]
    array = np.array(raw, dtype=float)
    return array[:, :2], array[:, 2].astype(int), "classification", ["bad", "good"]


def _load_sklearn_dataset(loader_name: str) -> tuple[np.ndarray, np.ndarray, str, list[str]]:
    """Iris / diabetes come from scikit-learn; it is an optional dependency."""
    try:
        import sklearn.datasets  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(f"dataset '{loader_name}' requires scikit-learn: pip install scikit-learn") from exc
    data = getattr(sklearn.datasets, loader_name)()
    task_type = "regression" if loader_name == "load_diabetes" else "classification"
    target_names = list(data.target_names) if hasattr(data, "target_names") else []
    return data.data, data.target, task_type, target_names


SAMPLE_DATASETS = {
    "watermelon": _load_watermelon,
    "iris": lambda: _load_sklearn_dataset("load_iris"),
    "diabetes": lambda: _load_sklearn_dataset("load_diabetes"),
}


def list_sample_datasets() -> list[str]:
    return sorted(SAMPLE_DATASETS)


def load_sample_dataset(name: str) -> tuple[np.ndarray, np.ndarray, str, list[str]]:
    """Return (X, y, task_type, target_names) for a built-in sample dataset."""
    if name not in SAMPLE_DATASETS:
        raise ValueError(f"unknown sample dataset: {name}; available: {sorted(SAMPLE_DATASETS)}")
    return SAMPLE_DATASETS[name]()
