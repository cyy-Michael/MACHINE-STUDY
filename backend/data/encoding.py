# -*- coding: utf-8 -*-
"""缺失值、类型推断与特征编码。

**防泄露是这里的核心设计**：``build_design_matrix`` 故意保留 ``np.nan``
而不做填充，因为均值/中位数填充必须只用训练集拟合。填充由
``impute_with_train_statistics`` 在 ``train_test_split`` 之后完成，
与 ``preprocessing.standardize`` 的「统计量只来自训练集」保持一致。
"""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

# 各数据集的缺失标记并不统一：credit 用 "?"，AirQuality 用 -200，
# 空白单元格则一律算缺失。具体取值由 metadata 逐数据集声明。
DEFAULT_MISSING_MARKERS = ("", "?", "NA", "NaN", "nan", "null", "NULL", "None")

# 一列中可解析为数字的比例低于该阈值就判定为类别列。
NUMERIC_RATIO_THRESHOLD = 0.95

# 类别列基数超过该值时改用整数编码，避免 one-hot 把维度炸开。
MAX_ONEHOT_LEVELS = 50

MISSING_LEVEL = "<缺失>"


class DesignMatrix(NamedTuple):
    X: np.ndarray
    feature_names: list[str]
    y_raw: list[str]
    column_types: dict[str, str]
    categorical_levels: dict[str, list[str]]
    coerced_missing: dict[str, int]


# --------------------------------------------------------------------------
# 缺失标记
# --------------------------------------------------------------------------

def normalize_cell(cell: str, decimal_comma: bool = False) -> str:
    """规范化单元格文本，便于与缺失标记比较。

    AirQualityUCI 的列用逗号当小数点，同一个缺失值会写成 ``-200`` 或
    ``-200,0``。只做原始文本比较会漏判 —— 实测该数据集的目标列正是这样
    漏进了 -200，把回归目标的取值范围污染成 ``-200..63.7``。
    """
    text = cell.strip()
    return text.replace(",", ".") if decimal_comma else text


def missing_marker_set(
    markers: tuple[str, ...] | list[str],
    decimal_comma: bool = False,
) -> frozenset[str]:
    """把缺失标记展开成集合，并加入它们的数值等价写法。

    ``-200`` 会同时匹配 ``-200``、``-200.0``、``-200,0``，这样小数点写法
    和多余的 ``.0`` 都绕不过缺失判断。非数值标记（如 ``?``）原样保留。
    """
    result: set[str] = set()
    for marker in markers:
        text = marker.strip()
        result.add(text)
        result.add(text.replace(",", "."))
        try:
            value = float(text.replace(",", "."))
        except ValueError:
            continue
        result.add(repr(value))
        result.add(str(value))
    return frozenset(result)


def is_missing(
    cell: str,
    marker_set: frozenset[str] | set[str],
    decimal_comma: bool = False,
) -> bool:
    return normalize_cell(cell, decimal_comma) in marker_set


def parse_float(text: str, decimal_comma: bool = False) -> float:
    """把单元格转成 float。

    ``decimal_comma=True`` 对应 AirQualityUCI：那里逗号是小数点（``2,6`` = 2.6），
    且不存在千分位分隔符，所以只需把 ``','`` 换成 ``'.'``。
    """
    return float(normalize_cell(text, decimal_comma))


# --------------------------------------------------------------------------
# 类型推断
# --------------------------------------------------------------------------

def infer_column_types(
    header: list[str],
    rows: list[list[str]],
    *,
    missing_markers: tuple[str, ...] | list[str] = DEFAULT_MISSING_MARKERS,
    decimal_comma: bool = False,
    dtypes: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, int]]:
    """推断每列是 ``numeric`` 还是 ``nominal``。

    metadata 里显式声明的 ``dtypes`` 优先，不会被推断覆盖。
    返回 ``(类型表, 被强制当作缺失的单元格数)``——后者是数据质量证据：
    声明为数值却解析失败的单元格会被计到这里，而不是静默变成 0。
    """
    dtypes = dtypes or {}
    markers = missing_marker_set(missing_markers, decimal_comma)
    types: dict[str, str] = {}
    coerced: dict[str, int] = {}

    for index, name in enumerate(header):
        if name in dtypes:
            types[name] = dtypes[name]
            coerced[name] = 0
            continue

        values = [row[index] if index < len(row) else "" for row in rows]
        present = [v for v in values if not is_missing(v, markers, decimal_comma)]
        if not present:
            types[name] = "nominal"
            coerced[name] = 0
            continue

        numeric = 0
        for value in present:
            try:
                parse_float(value, decimal_comma)
                numeric += 1
            except ValueError:
                continue

        ratio = numeric / len(present)
        if ratio >= NUMERIC_RATIO_THRESHOLD:
            types[name] = "numeric"
            coerced[name] = len(present) - numeric
        else:
            types[name] = "nominal"
            coerced[name] = 0

    return types, coerced


# --------------------------------------------------------------------------
# 设计矩阵
# --------------------------------------------------------------------------

def build_design_matrix(
    header: list[str],
    rows: list[list[str]],
    *,
    target: str,
    dtypes: dict[str, str] | None = None,
    missing_markers: tuple[str, ...] | list[str] = DEFAULT_MISSING_MARKERS,
    decimal_comma: bool = False,
    drop_columns: tuple[str, ...] | list[str] = (),
) -> DesignMatrix:
    """把字符串表变成数值设计矩阵，缺失的数值保留为 ``np.nan``。

    * 数值列 → 一列 float，缺失处为 ``np.nan``
    * 类别列 → one-hot，并把缺失单列为一个 ``<缺失>`` 类别
      （丢掉缺失样本会让 ckd 损失 10% 数据，不划算）
    """
    if target not in header:
        raise ValueError(f"目标列 {target!r} 不在表头中：{header}")

    drop = set(drop_columns)
    markers = missing_marker_set(missing_markers, decimal_comma)
    types, coerced = infer_column_types(
        header, rows, missing_markers=missing_markers,
        decimal_comma=decimal_comma, dtypes=dtypes,
    )

    blocks: list[np.ndarray] = []
    names: list[str] = []
    levels_map: dict[str, list[str]] = {}

    for index, name in enumerate(header):
        if name == target or name in drop:
            continue
        column = [row[index] if index < len(row) else "" for row in rows]

        if types[name] == "numeric":
            values = np.empty(len(column), dtype=float)
            for position, cell in enumerate(column):
                if is_missing(cell, markers, decimal_comma):
                    values[position] = np.nan
                    continue
                try:
                    values[position] = parse_float(cell, decimal_comma)
                except ValueError:
                    # 声明为 numeric 却解析失败：按缺失处理，计数已在 coerced 里
                    values[position] = np.nan
            blocks.append(values.reshape(-1, 1))
            names.append(name)
            continue

        present_levels = sorted(
            {normalize_cell(c, decimal_comma) for c in column
             if not is_missing(c, markers, decimal_comma)}
        )
        normalized = [normalize_cell(c, decimal_comma) for c in column]

        if len(present_levels) > MAX_ONEHOT_LEVELS:
            # 高基数类别退化为整数编码，避免维度爆炸
            mapping = {level: code for code, level in enumerate(present_levels)}
            codes = np.array(
                [mapping.get(cell, -1) for cell in normalized], dtype=float
            ).reshape(-1, 1)
            blocks.append(codes)
            names.append(name)
            levels_map[name] = present_levels
            continue

        for level in present_levels:
            indicator = np.array([1.0 if cell == level else 0.0 for cell in normalized])
            blocks.append(indicator.reshape(-1, 1))
            names.append(f"{name}={level}")
        # 缺失自成一类：该列有缺失时才有这一列
        if any(is_missing(c, markers, decimal_comma) for c in column):
            indicator = np.array(
                [1.0 if is_missing(c, markers, decimal_comma) else 0.0 for c in column]
            )
            blocks.append(indicator.reshape(-1, 1))
            names.append(f"{name}={MISSING_LEVEL}")
        levels_map[name] = present_levels

    if not blocks:
        raise ValueError("没有任何特征列可用（是否把全部列都 drop 了？）")

    X = np.column_stack(blocks)
    target_index = header.index(target)
    y_raw = [row[target_index] if target_index < len(row) else "" for row in rows]

    return DesignMatrix(
        X=X,
        feature_names=names,
        y_raw=y_raw,
        column_types={name: types[name] for name in header},
        categorical_levels=levels_map,
        coerced_missing=coerced,
    )


# --------------------------------------------------------------------------
# 目标编码
# --------------------------------------------------------------------------

def encode_target(
    y_raw: list[str],
    task_type: str,
    *,
    missing_markers: tuple[str, ...] | list[str] = DEFAULT_MISSING_MARKERS,
    decimal_comma: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """回归返回 float 向量；分类返回整数编码 + 标签名（供混淆矩阵使用）。

    调用方应先用 :func:`target_row_mask` 剔除目标缺失的样本，这里只做兜底校验。
    """
    markers = missing_marker_set(missing_markers, decimal_comma)

    if task_type == "regression":
        values = np.empty(len(y_raw), dtype=float)
        for position, cell in enumerate(y_raw):
            if is_missing(cell, markers, decimal_comma):
                raise ValueError(
                    f"目标列存在缺失值（第 {position} 行）；应先调用 target_row_mask 剔除"
                )
            values[position] = parse_float(cell, decimal_comma)
        return values, []

    if task_type not in ("classification", "clustering"):
        raise ValueError(f"不支持的 task_type：{task_type}")

    if any(is_missing(cell, markers, decimal_comma) for cell in y_raw):
        raise ValueError("目标列存在缺失值，无法用于分类")

    levels = sorted({normalize_cell(cell, decimal_comma) for cell in y_raw})
    mapping = {level: code for code, level in enumerate(levels)}
    return np.array([mapping[normalize_cell(c, decimal_comma)] for c in y_raw], dtype=int), levels


def target_row_mask(
    header: list[str],
    rows: list[list[str]],
    target: str,
    *,
    missing_markers: tuple[str, ...] | list[str] = DEFAULT_MISSING_MARKERS,
    decimal_comma: bool = False,
) -> list[bool]:
    """标出目标列非缺失的行。

    目标缺失的样本对监督学习毫无价值（既不能训练也不能评估），应当整行剔除
    并计数上报，而不是抛异常让整个数据集不可用。
    """
    markers = missing_marker_set(missing_markers, decimal_comma)
    index = header.index(target)
    return [
        not is_missing(row[index] if index < len(row) else "", markers, decimal_comma)
        for row in rows
    ]


# --------------------------------------------------------------------------
# 缺失值填充（只在训练集上拟合）
# --------------------------------------------------------------------------

def impute_with_train_statistics(
    X_train: np.ndarray,
    X_test: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """用**训练集列均值**填充 ``np.nan``；整列全缺失时填 0。

    测试集用的是训练集的均值——这是防泄露的关键，不能用测试集自己的统计量。
    """
    X_train = np.asarray(X_train, dtype=float)
    if X_train.ndim != 2:
        raise ValueError("X_train 必须是二维数组")

    means = np.nanmean(X_train, axis=0)
    means = np.where(np.isnan(means), 0.0, means)

    def fill(matrix: np.ndarray) -> np.ndarray:
        filled = np.array(matrix, dtype=float, copy=True)
        missing = np.isnan(filled)
        if missing.any():
            filled[missing] = np.take(means, np.where(missing)[1])
        return filled

    if X_test is None:
        return fill(X_train), means
    return fill(X_train), fill(np.asarray(X_test, dtype=float)), means


# --------------------------------------------------------------------------
# 数据质量报告
# --------------------------------------------------------------------------

def missing_report(
    header: list[str],
    rows: list[list[str]],
    missing_markers: tuple[str, ...] | list[str] = DEFAULT_MISSING_MARKERS,
    decimal_comma: bool = False,
) -> dict[str, int]:
    """统计每列缺失单元格数，供「接口二」返回数据质量信息。"""
    markers = missing_marker_set(missing_markers, decimal_comma)
    report: dict[str, int] = {}
    for index, name in enumerate(header):
        report[name] = sum(
            1
            for row in rows
            if is_missing(row[index] if index < len(row) else "", markers, decimal_comma)
        )
    return report


def dataset_summary(matrix: DesignMatrix) -> dict[str, Any]:
    """给接口二用的一句话统计。"""
    return {
        "n_features": int(matrix.X.shape[1]),
        "n_samples": int(matrix.X.shape[0]),
        "column_types": matrix.column_types,
        "coerced_missing": matrix.coerced_missing,
    }


# --------------------------------------------------------------------------
# 重复行
# --------------------------------------------------------------------------

def row_signature(row: list[str]) -> tuple[str, ...]:
    """整行的比较键。去空白后再比，避免 "a" 与 " a " 被当成两行。"""
    return tuple(cell.strip() for cell in row)


def duplicate_row_count(rows: list[list[str]]) -> int:
    """完全重复的行数。

    这不是理论问题：实测 phishing 有 5206/11055 行重复（47%），iris 3 行，
    concrete 25 行。重复行会让同一行同时落进训练集与测试集，指标虚高。
    """
    seen: set[tuple[str, ...]] = set()
    duplicates = 0
    for row in rows:
        key = row_signature(row)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


def deduplicate_rows(rows: list[list[str]]) -> tuple[list[list[str]], int]:
    """按整行去重，保留首次出现，返回 ``(去重后的行, 被删掉的行数)``。"""
    seen: set[tuple[str, ...]] = set()
    unique: list[list[str]] = []
    for row in rows:
        key = row_signature(row)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique, len(rows) - len(unique)
