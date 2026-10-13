# -*- coding: utf-8 -*-
"""测试共用的跳过条件。

数据集本体不进仓库（``.gitignore`` 排除了 ``数据集/``），所以依赖真实数据的
用例在别的机器上必须**优雅跳过**而不是报错——否则 README 里说的
「按说明就能跑起来」就成了一句空话。
"""
from __future__ import annotations

import unittest

from backend.data.registry import dataset_names, dataset_path, get_spec


def missing_datasets() -> list[str]:
    """列出本地文件缺失的数据集。"""
    missing: list[str] = []
    for name in dataset_names():
        try:
            path = dataset_path(get_spec(name))
        except Exception:  # noqa: BLE001 - 元数据坏了也应该算缺失
            missing.append(name)
            continue
        if not path.exists():
            missing.append(name)
    return missing


def dataset_available(name: str) -> bool:
    try:
        return dataset_path(get_spec(name)).exists()
    except Exception:  # noqa: BLE001
        return False


_MISSING = missing_datasets()
HAS_DATA = bool(dataset_names()) and not _MISSING

# 需要「全部 8 个数据集都在本地」的用例
requires_data = unittest.skipUnless(
    HAS_DATA,
    "本地缺少数据集文件（数据集本体不进仓库，已跳过）。"
    "把数据放到仓库根的 数据集/ 目录，或设置环境变量 ML_DATA_ROOT 指向数据目录。",
)


def requires_dataset(name: str):
    """只依赖某一个数据集的用例。"""
    return unittest.skipUnless(
        dataset_available(name),
        f"本地缺少数据集 {name!r}（数据集本体不进仓库，已跳过）。",
    )
