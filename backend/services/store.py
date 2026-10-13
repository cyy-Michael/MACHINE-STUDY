# -*- coding: utf-8 -*-
"""实验记录落盘。

写到 ``experiments/results/`` 下的 JSON（该目录已被 ``.gitignore`` 排除），
一次实验一个文件。

为什么用文件而不是 SQLite：这个项目的实验量是「手动点几十次」级别，JSON
便于直接打开查看、也便于写进报告；而且不引入额外依赖。真到了需要并发查询
的量级，把本模块的实现换成 SQLite 即可，对外函数签名不用动。
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from backend.logging_config import get_logger

logger = get_logger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "results"

# experiment_id 会直接拼进文件名，必须挡住路径分隔符，防止跳出目录
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,200}$")


class ExperimentNotFound(KeyError):
    """本地没有这条实验记录。"""


def results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def _path_for(experiment_id: str) -> Path:
    if not _SAFE_ID.match(experiment_id):
        raise ValueError(f"非法 experiment_id：{experiment_id!r}")
    return results_dir() / f"{experiment_id}.json"


def save_experiment(result: dict[str, Any]) -> Path:
    """原子落盘：先写临时文件再 ``os.replace``。

    直接写目标文件的话，并发读可能读到写了一半的 JSON。
    """
    experiment_id = result.get("experiment_id")
    if not experiment_id:
        raise ValueError("结果里缺少 experiment_id")
    path = _path_for(str(experiment_id))
    payload = json.dumps(result, ensure_ascii=False, indent=2)

    handle_fd, temp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        # 落盘失败要能看到是哪条实验、写到哪个路径失败的
        logger.error("实验记录写入失败 id=%s path=%s", experiment_id, path, exc_info=True)
        raise
    logger.debug("实验记录已保存 id=%s -> %s", experiment_id, path)
    return path


def load_experiment(experiment_id: str) -> dict[str, Any]:
    path = _path_for(experiment_id)
    if not path.exists():
        raise ExperimentNotFound(f"找不到实验记录 {experiment_id!r}")
    return json.loads(path.read_text(encoding="utf-8"))


def experiment_exists(experiment_id: str) -> bool:
    try:
        return _path_for(experiment_id).exists()
    except ValueError:
        return False


def list_experiments(limit: int = 50) -> list[dict[str, Any]]:
    """按时间倒序列出实验摘要（不返回预测值与散点，避免响应过大）。"""
    directory = results_dir()
    # 文件名以时间戳开头，所以按名字倒序即按时间倒序
    files = sorted(directory.glob("*.json"), reverse=True)

    summaries: list[dict[str, Any]] = []
    for path in files[: max(0, limit)]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # 单个坏文件不该让整个列表接口失败，但必须留下痕迹
            logger.warning("跳过损坏的实验记录文件：%s", path, exc_info=True)
            continue
        summaries.append(
            {
                "experiment_id": payload.get("experiment_id", path.stem),
                "status": payload.get("status", "unknown"),
                "dataset": payload.get("dataset"),
                "algorithm": payload.get("algorithm"),
                "task_type": payload.get("task_type"),
                "parameters": payload.get("parameters"),
                "metrics": payload.get("metrics"),
                "train_time_seconds": payload.get("train_time_seconds"),
                "predict_time_seconds": payload.get("predict_time_seconds"),
                "created_at": payload.get("created_at"),
                "error": payload.get("error"),
                "saved": True,
            }
        )
    return summaries


def delete_experiment(experiment_id: str) -> bool:
    path = _path_for(experiment_id)
    if not path.exists():
        return False
    path.unlink()
    return True
