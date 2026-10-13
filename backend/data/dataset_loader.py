"""Dataset loading adapters for the local files in backend/data/local.

Each adapter returns (X, y, metadata) where metadata describes columns,
task type and label column so the API can build dataset detail responses.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

DATA_LOCAL = Path(__file__).resolve().parent.parent / "data" / "local"


def _resolve_path(name: str) -> Path:
    for path in DATA_LOCAL.iterdir():
        if path.is_file() and path.stem == name:
            return path
    raise FileNotFoundError(f"dataset '{name}' not found in {DATA_LOCAL}")


def load_iris(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Iris: comma CSV, no header, last column is the string label."""
    records = np.genfromtxt(path, delimiter=",", dtype=str)
    X = records[:, :-1].astype(float)
    y = records[:, -1]
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    return X, y, {
        "name": path.stem,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "task_type": "classification",
        "label_column": "species",
        "columns": columns,
    }


def load_phishing(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Phishing Websites: ARFF, last attribute 'Result' is the label."""
    attributes: list[str] = []
    data_rows: list[list[float]] = []
    in_data = False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("@attribute"):
                # @attribute name { values }  or  @attribute name numeric
                parts = line.split()
                attributes.append(parts[1])
            elif line.lower() == "@data":
                in_data = True
            elif in_data:
                data_rows.append([float(v) for v in line.split(",")])
    array = np.array(data_rows, dtype=float)
    X = array[:, :-1]
    y = array[:, -1].astype(int).astype(str)  # label is -1 / 1
    return X, y, {
        "name": path.stem,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "task_type": "classification",
        "label_column": attributes[-1],
        "columns": attributes,
    }


def load_air_quality(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Air Quality: semicolon CSV, European decimals (comma), label 'C6H6(GT)'."""
    import csv

    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        for row in reader:
            if row:
                rows.append(row)

    def to_float(value: str) -> float:
        # European decimal: "2,6" -> 2.6 ; empty / missing -> -200
        v = value.strip()
        if v == "":
            return -200.0
        return float(v.replace(",", "."))

    # strip trailing empty columns
    n_cols = len(header)
    while n_cols > 0 and header[n_cols - 1].strip() == "":
        n_cols -= 1
    header = header[:n_cols]
    rows = [r[:n_cols] for r in rows]

    # convert numeric columns, skip Date/Time
    numeric_cols = [i for i, h in enumerate(header) if h not in ("Date", "Time")]
    matrix = np.array(
        [[to_float(r[i]) for i in numeric_cols] for r in rows if len(r) >= n_cols],
        dtype=float,
    )
    label_idx = header.index("C6H6(GT)")
    feature_cols = [i for i in numeric_cols if i != label_idx]
    X = matrix[:, [numeric_cols.index(i) for i in feature_cols]]
    y = matrix[:, numeric_cols.index(label_idx)]
    return X, y, {
        "name": path.stem,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "task_type": "regression",
        "label_column": "C6H6(GT)",
        "columns": header,
    }


_LOADERS = {
    "iris": load_iris,
    "phishing": load_phishing,
    "air_quality": load_air_quality,
}


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a local dataset by name and return (X, y, metadata)."""
    if name not in _LOADERS:
        raise ValueError(f"unknown dataset: {name}; available: {sorted(_LOADERS)}")
    path = _resolve_path(name)
    return _LOADERS[name](path)


def dataset_preview(name: str) -> dict[str, Any]:
    """Return the detail object the API serves for GET /api/datasets/{name}."""
    X, y, meta = load_dataset(name)
    head: list[dict[str, Any]] = []
    n_preview = min(5, X.shape[0])
    feat_cols = meta["columns"][:-1] if meta["task_type"] == "classification" else [
        c for c in meta["columns"] if c not in ("Date", "Time", meta["label_column"])
    ]
    for i in range(n_preview):
        row = {col: float(X[i, j]) for j, col in enumerate(feat_cols)}
        row[meta["label_column"]] = y[i].item() if hasattr(y[i], "item") else y[i]
        head.append(row)
    # label distribution
    labels, counts = np.unique(y, return_counts=True)
    label_dist = {str(lbl): int(cnt) for lbl, cnt in zip(labels, counts)}
    return {**meta, "head": head, "label_distribution": label_dist}
