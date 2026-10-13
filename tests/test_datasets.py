# -*- coding: utf-8 -*-
"""数据集登记表测试。

分两层：
* **元数据结构校验**不需要数据文件，任何机器上都能跑，防止元数据与列名漂移；
* **实际加载校验**需要本地有数据，缺数据时自动跳过。
"""
from __future__ import annotations

import unittest

import numpy as np

from backend.algorithms.registry import ALGORITHM_REGISTRY
from backend.data.registry import (
    dataset_names,
    describe_dataset,
    get_spec,
    list_datasets,
    load_dataset,
)
from tests.support import requires_data

VALID_TASKS = {"classification", "regression"}


class MetadataStructureTest(unittest.TestCase):
    """只读 metadata JSON，不碰数据文件。"""

    def test_at_least_seven_datasets_registered(self):
        self.assertGreaterEqual(len(dataset_names()), 7)

    def test_name_matches_file_and_task_is_valid(self):
        for name in dataset_names():
            spec = get_spec(name)
            with self.subTest(dataset=name):
                self.assertEqual(spec.name, name)
                self.assertIn(spec.task_type, VALID_TASKS)

    def test_target_is_one_of_the_columns(self):
        for name in dataset_names():
            spec = get_spec(name)
            with self.subTest(dataset=name):
                self.assertIn(spec.target, spec.columns)

    def test_columns_are_unique_and_non_empty(self):
        for name in dataset_names():
            spec = get_spec(name)
            with self.subTest(dataset=name):
                self.assertTrue(spec.columns)
                self.assertEqual(len(spec.columns), len(set(spec.columns)))

    def test_dtypes_cover_every_column(self):
        for name in dataset_names():
            spec = get_spec(name)
            with self.subTest(dataset=name):
                self.assertEqual(set(spec.columns) - set(spec.dtypes), set())

    def test_drop_columns_exist_in_columns(self):
        for name in dataset_names():
            spec = get_spec(name)
            with self.subTest(dataset=name):
                for column in spec.drop_columns:
                    self.assertIn(column, spec.columns)

    def test_dropped_target_is_not_allowed(self):
        for name in dataset_names():
            spec = get_spec(name)
            with self.subTest(dataset=name):
                self.assertNotIn(spec.target, spec.drop_columns)

    def test_n_features_matches_column_arithmetic(self):
        for name in dataset_names():
            spec = get_spec(name)
            with self.subTest(dataset=name):
                expected = len(spec.columns) - 1 - len(spec.drop_columns)
                self.assertEqual(spec.n_features, expected)

    def test_recommended_algorithms_exist(self):
        for name in dataset_names():
            spec = get_spec(name)
            for algorithm in spec.recommended_algorithms:
                with self.subTest(dataset=name, algorithm=algorithm):
                    self.assertIn(algorithm, ALGORITHM_REGISTRY)

    def test_recommended_algorithms_match_task(self):
        """分类数据集不该推荐纯回归算法，反之亦然。"""
        for name in dataset_names():
            spec = get_spec(name)
            for algorithm in spec.recommended_algorithms:
                declared = ALGORITHM_REGISTRY[algorithm]["task_type"]
                if declared in ("clustering", "dimensionality_reduction"):
                    continue
                with self.subTest(dataset=name, algorithm=algorithm):
                    self.assertIn(spec.task_type, declared)

    def test_missing_markers_contain_blank(self):
        for name in dataset_names():
            spec = get_spec(name)
            with self.subTest(dataset=name):
                self.assertIn("", spec.missing_markers)

    def test_list_datasets_filtering(self):
        everything = list_datasets()
        self.assertEqual(len(everything), len(dataset_names()))

        regression = list_datasets(task_type="regression")
        self.assertTrue(regression)
        for item in regression:
            self.assertEqual(item["task_type"], "regression")

        self.assertEqual(list_datasets(task_type="nonexistent"), [])


@requires_data
class DatasetLoadingTest(unittest.TestCase):
    """需要本地数据文件。"""

    def test_every_dataset_loads_with_declared_shape(self):
        for name in dataset_names():
            loaded = load_dataset(name)
            with self.subTest(dataset=name):
                self.assertEqual(loaded.X.shape[0], loaded.spec.n_samples)
                self.assertEqual(len(loaded.y), loaded.spec.n_samples)
                self.assertEqual(loaded.X.shape[1], len(loaded.feature_names))
                self.assertTrue(np.isfinite(loaded.X).any() or loaded.X.size == 0)

    def test_feature_names_are_unique(self):
        for name in dataset_names():
            loaded = load_dataset(name)
            with self.subTest(dataset=name):
                self.assertEqual(len(loaded.feature_names), len(set(loaded.feature_names)))

    def test_no_infinite_values(self):
        """缺失必须是 np.nan；inf 会让算法层直接抛错。"""
        for name in dataset_names():
            loaded = load_dataset(name)
            with self.subTest(dataset=name):
                self.assertFalse(np.isinf(loaded.X).any())

    def test_target_is_never_nan(self):
        for name in dataset_names():
            loaded = load_dataset(name)
            with self.subTest(dataset=name):
                if loaded.spec.task_type == "regression":
                    self.assertTrue(np.isfinite(loaded.y).all())
                else:
                    self.assertGreaterEqual(int(loaded.y.min()), 0)

    def test_classification_label_names_match_target_values(self):
        for name in dataset_names():
            loaded = load_dataset(name)
            if loaded.spec.task_type != "classification":
                continue
            with self.subTest(dataset=name):
                self.assertEqual(set(loaded.y.tolist()), set(range(len(loaded.label_names))))

    def test_describe_and_load_agree_on_sample_count(self):
        """接口二与训练必须用同一套口径。

        否则会出现「详情页说 150 行、实际只训练 147 行」这种前端对不上数的
        问题。去重与目标缺失剔除都要在两个入口保持一致。
        """
        for name in dataset_names():
            loaded = load_dataset(name)
            described = describe_dataset(name)
            with self.subTest(dataset=name):
                self.assertEqual(described["n_samples"], loaded.X.shape[0])
                self.assertEqual(
                    described["duplicate_rows"], loaded.dropped_duplicates
                )
                self.assertEqual(
                    described["dropped_target_missing_rows"], loaded.dropped_rows
                )

    def test_describe_dataset_shape(self):
        payload = describe_dataset("iris")
        self.assertEqual(payload["name"], "iris")
        self.assertEqual(len(payload["fields"]), len(get_spec("iris").columns))
        self.assertIn("label_distribution", payload)
        self.assertEqual(
            sum(payload["label_distribution"].values()), payload["n_samples"]
        )

    def test_iris_matches_documented_distribution(self):
        """iris 文件里 150 行含 3 行完全重复，去重后是 48/50/49。

        这 3 个重复行的标签与原行一致，没有标签冲突，所以去重是安全的。
        """
        loaded = load_dataset("iris")
        self.assertEqual(loaded.dropped_duplicates, 3)
        counts = np.bincount(loaded.y)
        np.testing.assert_array_equal(counts, [48, 50, 49])

    def test_duplicate_rows_are_dropped(self):
        """重复行会让同一行同时落进训练集和测试集，导致指标虚高。"""
        expected = {"iris": 3, "concrete": 25, "phishing": 5206}
        for name, duplicates in expected.items():
            loaded = load_dataset(name)
            with self.subTest(dataset=name):
                self.assertEqual(loaded.dropped_duplicates, duplicates)

    def test_datasets_without_duplicates_report_zero(self):
        for name in ("car", "ckd", "credit", "bike_day"):
            loaded = load_dataset(name)
            with self.subTest(dataset=name):
                self.assertEqual(loaded.dropped_duplicates, 0)

    def test_describe_reports_duplicate_count(self):
        payload = describe_dataset("phishing")
        self.assertEqual(payload["duplicate_rows"], 5206)
        self.assertTrue(payload["drop_duplicates"])

    def test_ckd_rows_with_extra_comma_are_repaired(self):
        """ckd 有 3 行字段数多一个，修复后类别分布应是文档中的 250/150。"""
        loaded = load_dataset("ckd")
        self.assertEqual(loaded.table.repaired, 3)
        counts = dict(zip(loaded.label_names, np.bincount(loaded.y).tolist()))
        self.assertEqual(counts, {"ckd": 250, "notckd": 150})

    def test_airquality_target_has_no_missing_marker_leakage(self):
        """-200 是缺失标记；它绝不能出现在回归目标里。"""
        loaded = load_dataset("airquality")
        self.assertGreater(loaded.dropped_rows, 0)
        self.assertGreaterEqual(float(loaded.y.min()), 0.0)
        self.assertLess(float(loaded.y.max()), 1000.0)

    def test_bike_day_drops_leaky_columns(self):
        """casual + registered == cnt，作为特征会造成标签泄露。

        注意断言的是 ``feature_names``（实际参与建模的列）而不是
        ``column_types`` —— 后者是整张表的类型图，包含目标列和被丢弃的列，
        那是有意保留的信息。
        """
        loaded = load_dataset("bike_day")
        features = set(loaded.feature_names)
        for leaky in ("casual", "registered", "instant", "dteday"):
            with self.subTest(column=leaky):
                self.assertNotIn(leaky, features)
                self.assertEqual(
                    [f for f in features if f.startswith(f"{leaky}=")], []
                )


if __name__ == "__main__":
    unittest.main()
