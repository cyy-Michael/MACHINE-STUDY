# -*- coding: utf-8 -*-
"""测试共用的跳过条件。

数据集本体不进仓库（``.gitignore`` 排除了 ``数据集/``），所以依赖真实数据的
用例在别的机器上必须**优雅跳过**而不是报错——否则 README 里说的
「按说明就能跑起来」就成了一句空话。
"""
from __future__ import annotations

import os
import unittest

# 测试期间把日志降到 WARNING：请求中间件会给每个用例打一行 INFO，
# 141 个用例的输出会被淹没，真正的问题反而看不见。
# 同时关掉文件输出，避免跑一次测试就在 logs/ 里留一堆文件。
os.environ.setdefault("ML_LOG_LEVEL", "WARNING")
os.environ.setdefault("ML_LOG_TO_FILE", "0")

from backend.data.registry import dataset_names, dataset_path, get_spec  # noqa: E402
from backend.logging_config import configure_logging  # noqa: E402

configure_logging(force=True)


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
