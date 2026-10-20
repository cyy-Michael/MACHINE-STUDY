# -*- coding: utf-8 -*-
"""统一日志配置。

设计取舍：

* **控制台 + 轮转文件双输出**。控制台给人看，``logs/backend.log`` 给排错用，
  单文件 2 MB、保留 3 个备份，不会把磁盘写满。
* **重复调用安全**。``uvicorn --reload`` 会重新导入模块，不加锁去重就会
  叠出好几份 handler，同一条日志打印好几遍。
* **Windows 控制台编码**。默认代码页是 GBK，日志里的中文和符号可能触发
  ``UnicodeEncodeError``。这在本项目里不是假想问题：根目录那个旧的
  ``test_registry.py`` 就是因为 print emoji 在 GBK 控制台崩掉，才让
  ``python -m unittest discover`` 整个套件失败的。所以这里显式放宽编码错误
  处理，**保证日志本身永远不会成为崩溃源**。
* **文件始终 UTF-8**，避免换台机器看日志变乱码。

环境变量：

* ``ML_LOG_LEVEL`` —— 日志级别，默认 ``INFO``（可选 DEBUG/INFO/WARNING/ERROR）
* ``ML_LOG_DIR``   —— 日志目录，默认仓库根的 ``logs/``
* ``ML_LOG_TO_FILE`` —— 设为 ``0`` 可关闭文件输出（测试时有用）
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO_ROOT / "logs"
LOG_FILE_NAME = "backend.log"

DEFAULT_LEVEL = "INFO"
DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)-32s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

MAX_BYTES = 2 * 1024 * 1024
BACKUP_COUNT = 3

# 我们自己的日志统一挂在这个 logger 下，方便与第三方库（uvicorn 等）分开控制
ROOT_LOGGER_NAME = "backend"

_lock = threading.Lock()
_configured = False


def log_directory() -> Path:
    configured = os.environ.get("ML_LOG_DIR")
    return Path(configured) if configured else DEFAULT_LOG_DIR


def log_level() -> str:
    return os.environ.get("ML_LOG_LEVEL", DEFAULT_LEVEL).upper()


def file_logging_enabled() -> bool:
    return os.environ.get("ML_LOG_TO_FILE", "1") not in ("0", "false", "False")


class _SafeStreamHandler(logging.StreamHandler):
    """即使控制台编码装不下某个字符，也只降级显示，绝不抛异常。"""

    def __init__(self, stream=None) -> None:
        stream = stream if stream is not None else sys.stderr
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                # 只放宽错误处理，不改编码 —— 改编码会让控制台显示变乱码
                reconfigure(errors="backslashreplace")
            except (ValueError, OSError):
                pass
        super().__init__(stream)


def configure_logging(
    level: str | None = None,
    *,
    log_to_file: bool | None = None,
    force: bool = False,
) -> logging.Logger:
    """配置并返回 ``backend`` logger。可重复调用。"""
    global _configured

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    with _lock:
        if _configured and not force:
            return logger
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # noqa: BLE001 - 关闭失败不该阻断启动
                pass
        _configured = True

    resolved_level = (level or log_level()).upper()
    logger.setLevel(getattr(logging, resolved_level, logging.INFO))
    # 不向 root 传播，避免和 uvicorn 的 handler 重复打印
    logger.propagate = False

    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DATE_FORMAT)

    console = _SafeStreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_to_file is None:
        log_to_file = file_logging_enabled()
    if log_to_file:
        try:
            directory = log_directory()
            directory.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                directory / LOG_FILE_NAME,
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as exc:
            # 磁盘只读或没权限时，退回只有控制台输出，不能让服务起不来
            logger.warning("无法写入日志文件（%s），本次仅输出到控制台", exc)

    return logger


def get_logger(name: str) -> logging.Logger:
    """取一个子 logger，首次调用时自动完成配置。

    ``name`` 建议传 ``__name__``，日志里就能看出是哪个模块产生的。
    """
    if not _configured:
        configure_logging()
    if name.startswith(ROOT_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
