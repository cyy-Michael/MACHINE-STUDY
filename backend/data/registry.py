# -*- coding: utf-8 -*-
"""数据集登记表。

元数据放在 ``backend/data/datasets/<name>.json``，本模块负责读取它、
解析文件路径、产出供 API 和训练编排使用的结构。

**数据集本体不进仓库**（``.gitignore`` 已排除 ``数据集/``），所以数据根目录
必须可配置：默认取仓库根的 ``数据集/``，也可用环境变量 ``ML_DATA_ROOT`` 覆盖。
换一台电脑只要按 README 放好数据即可。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

from .encoding import (
    build_design_matrix,
    encode_target,
    is_missing,
    missing_marker_set,
    missing_report,
    target_row_mask,
)
from .formats import Table, read_table

DATASET_DIR = Path(__file__).resolve().parent / "datasets"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "数据集"

# 登记名 -> 数据文件名，方便报错时给出可操作的提示
_HINT = "请确认数据集已放在 {root}，或设置环境变量 ML_DATA_ROOT 指向数据目录"


def data_root() -> Path:
    configured = os.environ.get("ML_DATA_ROOT")
    return Path(configured) if configured else DEFAULT_DATA_ROOT


class DatasetNotFound(KeyError):
    """登记表里没有这个数据集。"""


class DatasetUnavailable(RuntimeError):
    """登记了，但本地文件不存在或读不了。"""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    task_type: str
    target: str
    source: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    drop_columns: list[str] = field(default_factory=list)
    missing_markers: list[str] = field(default_factory=lambda: ["", "?"])
    decimal_comma: bool = False
    recommended_algorithms: list[str] = field(default_factory=list)
    shape: dict[str, int] = field(default_factory=dict)
    notes: str = ""

    @property
    def n_samples(self) -> int:
        return int(self.shape.get("n_samples", 0))

    @property
    def n_features(self) -> int:
        return int(self.shape.get("n_features", 0))


class LoadedDataset(NamedTuple):
    spec: DatasetSpec
    table: Table
    X: Any            # 未填充的设计矩阵，缺失处为 np.nan
    y: Any
    feature_names: list[str]
    label_names: list[str]
    column_types: dict[str, str]
    coerced_missing: dict[str, int]
    dropped_rows: int = 0   # 目标缺失而被剔除的样本数


# --------------------------------------------------------------------------
# 元数据读取
# --------------------------------------------------------------------------

def _spec_from_dict(payload: dict[str, Any]) -> DatasetSpec:
    known = {f for f in DatasetSpec.__dataclass_fields__}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"metadata 含未知字段 {sorted(unknown)}")
    return DatasetSpec(**payload)


@lru_cache(maxsize=1)
def _all_specs() -> dict[str, DatasetSpec]:
    specs: dict[str, DatasetSpec] = {}
    if not DATASET_DIR.exists():
        return specs
    for path in sorted(DATASET_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec = _spec_from_dict(payload)
        if spec.name != path.stem:
            raise ValueError(f"{path.name} 的 name 字段 {spec.name!r} 与文件名不一致")
        specs[spec.name] = spec
    return specs


def reload_specs() -> None:
    """清缓存，测试或热更新元数据后调用。"""
    _all_specs.cache_clear()
    _read_table_cached.cache_clear()


def get_spec(name: str) -> DatasetSpec:
    specs = _all_specs()
    if name not in specs:
        raise DatasetNotFound(f"未知数据集 {name!r}；可选：{sorted(specs)}")
    return specs[name]


def list_datasets(task_type: str | None = None) -> list[dict[str, Any]]:
    """接口一用：列出全部（或按任务类型筛选）数据集的摘要。

    只读元数据，不碰文件，所以列表接口是常数时间。
    """
    result = []
    for spec in _all_specs().values():
        if task_type and task_type != spec.task_type:
            continue
        result.append(
            {
                "name": spec.name,
                "display_name": spec.display_name,
                "task_type": spec.task_type,
                "n_samples": spec.n_samples,
                "n_features": spec.n_features,
                "target": spec.target,
                "description": spec.description,
                "source_format": spec.source.get("format", "csv"),
            }
        )
    return result


def dataset_names() -> list[str]:
    return sorted(_all_specs())


# --------------------------------------------------------------------------
# 文件读取
# --------------------------------------------------------------------------

def dataset_path(spec: DatasetSpec) -> Path:
    return data_root() / str(spec.source.get("path", ""))


def _source_key(source: dict[str, Any]) -> tuple:
    """构造可哈希的缓存键。

    **不能**把 value 一律 ``str()``：那会把 ``has_header`` 的布尔 ``False``
    变成字符串 ``"False"``，而 ``bool("False")`` 是 ``True`` —— 曾经因此
    让 iris / car / credit 三个无表头文件各自丢掉首行数据。
    """
    items: list[tuple[str, Any]] = []
    for key, value in source.items():
        if key == "path":
            continue
        if isinstance(value, (list, dict, set)):
            value = json.dumps(value, sort_keys=True, ensure_ascii=False)
        items.append((key, value))
    return tuple(sorted(items))


@lru_cache(maxsize=32)
def _read_table_cached(path_str: str, mtime: float, source_key: tuple, expected: int) -> Table:
    source = dict(source_key)
    return read_table(
        path_str,
        format=source.get("format", "csv"),
        delimiter=source.get("delimiter", ","),
        has_header=bool(source.get("has_header", False)),
        comment_prefix=source.get("comment_prefix"),
        sheet=source.get("sheet"),
        expected_columns=expected or None,
    )


def load_table(spec: DatasetSpec) -> Table:
    path = dataset_path(spec)
    if not path.exists():
        raise DatasetUnavailable(
            f"数据集 {spec.name!r} 的文件不存在：{path}\n"
            + _HINT.format(root=data_root())
        )
    source = spec.source
    table = _read_table_cached(
        str(path), path.stat().st_mtime, _source_key(source), len(spec.columns)
    )

    if spec.columns:
        # metadata 的 columns 是权威列名：无表头文件靠它命名，有表头文件靠它
        # 统一口径。列数对不上说明元数据已经和数据漂移，直接报错而不是硬跑。
        if len(spec.columns) != len(table.header):
            raise DatasetUnavailable(
                f"{spec.name}：metadata 声明 {len(spec.columns)} 列，"
                f"实际文件 {len(table.header)} 列 —— 元数据与数据不一致"
            )
        table = table._replace(header=list(spec.columns))
    return table


def load_raw(name: str) -> tuple[DatasetSpec, Table]:
    spec = get_spec(name)
    return spec, load_table(spec)


# --------------------------------------------------------------------------
# 设计矩阵
# --------------------------------------------------------------------------

def load_dataset(name: str) -> LoadedDataset:
    """加载并编码，**但故意不做缺失值填充**。

    填充和标准化必须只用训练集拟合，所以留给 ``services`` 层的编排函数
    在 ``train_test_split`` 之后完成（见 encoding.impute_with_train_statistics）。
    """
    spec, table = load_raw(name)
    missing_markers = tuple(spec.missing_markers)

    # 目标缺失的样本既不能训练也不能评估，整行剔除并计数上报。
    # AirQuality 的目标列存在 -200（写作 -200,0），若直接抛异常整个数据集就废了。
    mask = target_row_mask(
        table.header, table.rows, spec.target,
        missing_markers=missing_markers, decimal_comma=spec.decimal_comma,
    )
    rows = [row for row, keep in zip(table.rows, mask) if keep]
    dropped = len(table.rows) - len(rows)
    if not rows:
        raise DatasetUnavailable(f"{name}：目标列 {spec.target!r} 全部缺失，没有可用样本")

    matrix = build_design_matrix(
        table.header,
        rows,
        target=spec.target,
        dtypes=spec.dtypes,
        missing_markers=missing_markers,
        decimal_comma=spec.decimal_comma,
        drop_columns=spec.drop_columns,
    )
    y, label_names = encode_target(
        matrix.y_raw, spec.task_type,
        missing_markers=missing_markers, decimal_comma=spec.decimal_comma,
    )

    return LoadedDataset(
        spec=spec,
        table=table,
        X=matrix.X,
        y=y,
        feature_names=matrix.feature_names,
        label_names=label_names,
        column_types=matrix.column_types,
        coerced_missing=matrix.coerced_missing,
        dropped_rows=dropped,
    )


def describe_dataset(name: str) -> dict[str, Any]:
    """接口二用：字段列表、类型、样例、缺失统计、标签分布。"""
    spec, table = load_raw(name)
    missing_markers = tuple(spec.missing_markers)
    markers = missing_marker_set(missing_markers, spec.decimal_comma)
    missing = missing_report(
        table.header, table.rows, missing_markers, spec.decimal_comma
    )

    fields = []
    for index, column in enumerate(table.header):
        values = [row[index] if index < len(row) else "" for row in table.rows[:200]]
        present = [v for v in values if not is_missing(v, markers, spec.decimal_comma)]
        fields.append(
            {
                "name": column,
                "dtype": spec.dtypes.get(column) or ("target" if column == spec.target else "inferred"),
                "missing": missing.get(column, 0),
                "unique_sample": len(set(present)) if present else 0,
                "samples": list(dict.fromkeys(present))[:5],
            }
        )

    payload: dict[str, Any] = {
        "name": spec.name,
        "display_name": spec.display_name,
        "task_type": spec.task_type,
        "description": spec.description,
        "target": spec.target,
        "n_samples": len(table.rows),
        "n_features": max(0, len(table.header) - 1 - len(spec.drop_columns)),
        "source_format": table.format,
        "fields": fields,
        "missing_total": int(sum(missing.values())),
        "repaired_rows": table.repaired,
        "recommended_algorithms": spec.recommended_algorithms,
        "notes": spec.notes,
    }

    if spec.task_type == "classification":
        target_index = table.header.index(spec.target)
        distribution: dict[str, int] = {}
        for row in table.rows:
            label = row[target_index].strip() if target_index < len(row) else ""
            distribution[label] = distribution.get(label, 0) + 1
        payload["label_distribution"] = distribution

    return payload
