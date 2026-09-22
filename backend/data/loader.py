from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def load_csv(path: str | Path, *, delimiter: str = ",", has_header: bool = True) -> list[dict[str, Any]]:
    """Load a small CSV into records; richer format adapters belong in this package."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        if has_header:
            return list(csv.DictReader(file, delimiter=delimiter))
        return [{"column_" + str(i): value for i, value in enumerate(row)} for row in csv.reader(file, delimiter=delimiter)]


def list_local_datasets(root: str | Path = "data/local") -> list[dict[str, str]]:
    """Return dataset files without coupling the API to a particular file format."""
    root = Path(root)
    if not root.exists():
        return []
    supported = {".csv", ".txt", ".arff", ".xlsx", ".xls"}
    return [{"name": path.stem, "path": str(path), "format": path.suffix.lower().lstrip(".")} for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in supported]
