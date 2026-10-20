# -*- coding: utf-8 -*-
"""日志配置测试。

不依赖数据集，任何机器上都能跑。
"""
from __future__ import annotations

import logging
import os
import tempfile
import unittest
from logging.handlers import RotatingFileHandler
from pathlib import Path

import tests.support  # noqa: F401  —— 导入以套用测试期的日志降噪配置
from backend.logging_config import (
    ROOT_LOGGER_NAME,
    configure_logging,
    file_logging_enabled,
    get_logger,
    log_directory,
    log_level,
)


class LoggingConfigTest(unittest.TestCase):
    """注意：这些用例会改动全局日志状态，tearDown 里必须还原。

    ``tests.support`` 在导入时把级别降到 WARNING 并关掉文件输出，
    tearDown 重新调用 ``configure_logging(force=True)`` 就能回到那个状态。
    """

    _KEYS = ("ML_LOG_LEVEL", "ML_LOG_DIR", "ML_LOG_TO_FILE")

    def setUp(self) -> None:
        self._saved = {key: os.environ.get(key) for key in self._KEYS}

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        configure_logging(force=True)

    def test_configure_is_idempotent(self):
        """uvicorn --reload 会重复导入模块，不幂等就会叠出多份 handler。"""
        os.environ["ML_LOG_TO_FILE"] = "0"
        configure_logging("INFO", force=True)
        expected = len(logging.getLogger(ROOT_LOGGER_NAME).handlers)

        configure_logging("INFO")
        configure_logging("INFO")
        self.assertEqual(
            len(logging.getLogger(ROOT_LOGGER_NAME).handlers), expected
        )

    def test_force_replaces_handlers_instead_of_appending(self):
        os.environ["ML_LOG_TO_FILE"] = "0"
        configure_logging("INFO", force=True)
        for _ in range(5):
            configure_logging("INFO", force=True)
        self.assertEqual(len(logging.getLogger(ROOT_LOGGER_NAME).handlers), 1)

    def test_logger_is_namespaced_under_backend(self):
        self.assertEqual(get_logger("data.registry").name, "backend.data.registry")
        # 已经是全名的不再重复加前缀
        self.assertEqual(get_logger("backend.api").name, "backend.api")

    def test_propagate_is_off_to_avoid_double_printing(self):
        configure_logging("INFO", force=True)
        self.assertFalse(logging.getLogger(ROOT_LOGGER_NAME).propagate)

    def _release_file_handlers(self) -> None:
        """关掉文件 handler 并回到「不开文件日志」的状态。

        必须在退出 ``TemporaryDirectory`` 之前调用：Windows 上文件被占用就删不掉
        临时目录。而且**要先恢复环境变量**再重新配置 —— 否则 ``ML_LOG_DIR`` 还指向
        那个临时目录，重配时又会新建一个 handler 把文件重新占住。
        """
        os.environ["ML_LOG_TO_FILE"] = "0"
        os.environ.pop("ML_LOG_DIR", None)
        configure_logging("WARNING", force=True)

    def test_file_output_is_utf8_and_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["ML_LOG_DIR"] = tmp
            os.environ["ML_LOG_TO_FILE"] = "1"
            configure_logging("DEBUG", force=True)
            get_logger("test").info("中文日志不会乱码 %s", "✅")
            for handler in logging.getLogger(ROOT_LOGGER_NAME).handlers:
                handler.flush()

            path = Path(tmp) / "backend.log"
            self.assertTrue(path.exists())
            content = path.read_text(encoding="utf-8")
            self.assertIn("中文日志不会乱码", content)
            self.assertIn("backend.test", content)

            self._release_file_handlers()

    def test_file_logging_can_be_disabled(self):
        os.environ["ML_LOG_TO_FILE"] = "0"
        configure_logging("INFO", force=True)
        handlers = logging.getLogger(ROOT_LOGGER_NAME).handlers
        self.assertEqual(len(handlers), 1)
        self.assertFalse(
            any(isinstance(h, RotatingFileHandler) for h in handlers)
        )

    def test_file_handler_rotates(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["ML_LOG_DIR"] = tmp
            os.environ["ML_LOG_TO_FILE"] = "1"
            configure_logging("INFO", force=True)
            handler = next(
                h
                for h in logging.getLogger(ROOT_LOGGER_NAME).handlers
                if isinstance(h, RotatingFileHandler)
            )
            # 磁盘写满不该发生：必须有大小上限和备份数量
            self.assertGreater(handler.maxBytes, 0)
            self.assertGreater(handler.backupCount, 0)
            self._release_file_handlers()

    def test_level_falls_back_to_info_on_garbage(self):
        os.environ["ML_LOG_LEVEL"] = "NOT_A_LEVEL"
        configure_logging(force=True)
        # 不能因为写错级别就让整个服务起不来
        self.assertEqual(logging.getLogger(ROOT_LOGGER_NAME).level, logging.INFO)

    def test_unwritable_log_dir_does_not_break_startup(self):
        """日志目录不可写时只应降级为纯控制台输出，不能抛异常。

        用「拿一个已存在的文件当目录」来制造 mkdir 失败 —— 比塞非法路径可靠，
        Windows 上 ``os.environ`` 也不接受含空字符的值。
        """
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "not-a-directory"
            blocker.write_text("x", encoding="utf-8")
            os.environ["ML_LOG_DIR"] = str(blocker)
            os.environ["ML_LOG_TO_FILE"] = "1"
            configure_logging("INFO", force=True)

            handlers = logging.getLogger(ROOT_LOGGER_NAME).handlers
            self.assertEqual(len(handlers), 1, "应退回只有控制台输出")
            self.assertFalse(
                any(isinstance(h, RotatingFileHandler) for h in handlers)
            )

    def test_env_helpers(self):
        os.environ["ML_LOG_LEVEL"] = "debug"
        self.assertEqual(log_level(), "DEBUG")
        os.environ["ML_LOG_TO_FILE"] = "0"
        self.assertFalse(file_logging_enabled())
        os.environ["ML_LOG_TO_FILE"] = "1"
        self.assertTrue(file_logging_enabled())


class DefaultDestinationTest(unittest.TestCase):
    def test_default_log_dir_is_repo_logs(self):
        """默认落在仓库根的 logs/ 下。

        刻意不断言父目录叫什么 —— 换台机器克隆到别的文件夹名就挂了。
        """
        os.environ.pop("ML_LOG_DIR", None)
        directory = log_directory()
        self.assertTrue(directory.is_absolute())
        self.assertEqual(directory.name, "logs")


if __name__ == "__main__":
    unittest.main()
