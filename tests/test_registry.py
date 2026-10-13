# -*- coding: utf-8 -*-
"""算法注册表测试。

这个文件原先是仓库根目录的 ``test_registry.py``，靠 ``print`` 配 emoji 报告
结果。``python -m unittest discover``（不带 ``-s tests``）会把它当测试模块导入，
而 ✅/❌ 在 GBK 控制台会抛 ``UnicodeEncodeError``，把整个测试套件变成
``Ran 9 tests, FAILED (errors=1)``。

搬进 ``tests/`` 并改成真正的断言，顺带去掉了 emoji：
不再依赖控制台编码，也不再有「输出好看就算通过」的假测试。
"""
from __future__ import annotations

import unittest

from backend.algorithms.registry import create_algorithm, list_algorithms


class RegistryTest(unittest.TestCase):
    def test_list_algorithms_returns_ten_algorithms(self):
        algorithms = list_algorithms()
        self.assertEqual(len(algorithms), 10)

    def test_metadata_fields_are_complete(self):
        for item in list_algorithms():
            with self.subTest(algorithm=item.get("name")):
                for field in ("name", "display_name", "task_type", "implemented_in_house", "default_params"):
                    self.assertIn(field, item)
                # factory 是实现细节，不能出现在对外元数据里
                self.assertNotIn("factory", item)

    def test_task_type_filter(self):
        classification = list_algorithms(task_type="classification")
        self.assertTrue(classification)
        for item in classification:
            self.assertIn("classification", item["task_type"])

        # random_forest 与 gradient_boosting 的 task_type 是
        # "classification / regression"，所以两个字串筛选都会命中它们
        self.assertEqual(len(list_algorithms(task_type="regression")), 3)

    def test_unknown_task_type_returns_empty_list(self):
        self.assertEqual(list_algorithms(task_type="nonexistent"), [])

    def test_create_algorithm_returns_usable_model(self):
        model = create_algorithm("linear_regression")
        self.assertTrue(hasattr(model, "fit"))
        self.assertTrue(hasattr(model, "predict"))

    def test_create_unknown_algorithm_raises_value_error(self):
        with self.assertRaises(ValueError):
            create_algorithm("does_not_exist")


if __name__ == "__main__":
    unittest.main()
