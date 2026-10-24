# -*- coding: utf-8 -*-
"""编码层测试：缺失值识别、类型推断、one-hot、以及**防泄露**。

防泄露的用例是重点：项目文档把它列为评分点，而这类 bug 一旦出现，
模型指标会虚高且很难察觉。
"""
from __future__ import annotations

import unittest

import numpy as np

from backend.data.encoding import (
    build_design_matrix,
    deduplicate_rows,
    duplicate_row_count,
    encode_target,
    impute_with_train_statistics,
    infer_column_types,
    is_missing,
    missing_marker_set,
    missing_report,
    target_row_mask,
)
from backend.data.preprocessing import standardize


class MissingMarkerTest(unittest.TestCase):
    def test_marker_matches_numeric_equivalents(self):
        """``-200`` 必须同时匹配 ``-200,0`` 和 ``-200.0``。

        AirQualityUCI 用逗号做小数点，目标列的缺失值实际写作 ``-200,0``。
        只比原始文本会漏判，实测让回归目标的取值范围被污染成 ``-200..63.7``。
        """
        markers = missing_marker_set(["-200"], decimal_comma=True)
        for text in ("-200", "-200.0", "-200,0"):
            with self.subTest(text=text):
                self.assertTrue(is_missing(text, markers, decimal_comma=True))
        self.assertFalse(is_missing("-19,9", markers, decimal_comma=True))

    def test_question_mark_marker(self):
        markers = missing_marker_set(["", "?"])
        self.assertTrue(is_missing("?", markers))
        self.assertTrue(is_missing("   ", markers))
        self.assertFalse(is_missing("0", markers))

    def test_marker_set_ignores_non_numeric_markers(self):
        markers = missing_marker_set(["", "?", "NA"])
        self.assertIn("?", markers)
        self.assertIn("NA", markers)


class InferColumnTypesTest(unittest.TestCase):
    def test_numeric_and_nominal(self):
        header = ["num", "cat"]
        rows = [["1.5", "red"], ["2.5", "blue"], ["3.5", "red"]]
        types, coerced = infer_column_types(header, rows)
        self.assertEqual(types, {"num": "numeric", "cat": "nominal"})
        self.assertEqual(coerced, {"num": 0, "cat": 0})

    def test_declared_dtypes_win_over_inference(self):
        header = ["code"]
        rows = [["1"], ["2"], ["3"]]
        types, _ = infer_column_types(header, rows, dtypes={"code": "nominal"})
        self.assertEqual(types["code"], "nominal")

    def test_mostly_numeric_column_with_typo_stays_numeric(self):
        """少数字段解析失败时保留为数值列，并把失败数计数上报，而不是降级成类别列。"""
        header = ["x"]
        rows = [[str(i)] for i in range(100)] + [["oops"]]
        types, coerced = infer_column_types(header, rows)
        self.assertEqual(types["x"], "numeric")
        self.assertEqual(coerced["x"], 1)


class DesignMatrixTest(unittest.TestCase):
    def test_numeric_column_keeps_nan(self):
        header = ["x", "y"]
        rows = [["1.0", "a"], ["?", "b"], ["3.0", "a"]]
        matrix = build_design_matrix(header, rows, target="y")
        self.assertAlmostEqual(matrix.X[0, 0], 1.0)
        self.assertTrue(np.isnan(matrix.X[1, 0]))
        self.assertAlmostEqual(matrix.X[2, 0], 3.0)

    def test_nominal_column_becomes_one_hot_with_missing_category(self):
        header = ["color", "y"]
        rows = [["red", "a"], ["blue", "b"], ["?", "a"]]
        matrix = build_design_matrix(header, rows, target="y")
        self.assertEqual(
            matrix.feature_names,
            ["color=blue", "color=red", "color=<缺失>"],
        )
        np.testing.assert_allclose(matrix.X[0], [0, 1, 0])
        np.testing.assert_allclose(matrix.X[1], [1, 0, 0])
        np.testing.assert_allclose(matrix.X[2], [0, 0, 1])

    def test_drop_columns_are_excluded(self):
        header = ["keep", "leak", "y"]
        rows = [["1", "999", "0"], ["2", "999", "1"]]
        matrix = build_design_matrix(header, rows, target="y", drop_columns=["leak"])
        self.assertEqual(matrix.feature_names, ["keep"])

    def test_decimal_comma_and_missing_marker_together(self):
        """逗号小数点与 -200 缺失标记必须能同时生效。"""
        header = ["x", "target"]
        rows = [["2,6", "1,0"], ["-200,0", "2,0"], ["3,0", "-200,0"], ["1,0", "4,0"]]
        markers = ["", "?", "-200"]
        matrix = build_design_matrix(
            header, rows, target="target",
            missing_markers=markers, decimal_comma=True,
            dtypes={"x": "numeric", "target": "numeric"},
        )
        self.assertAlmostEqual(matrix.X[0, 0], 2.6)
        self.assertTrue(np.isnan(matrix.X[1, 0]))

        mask = target_row_mask(
            header, rows, "target", missing_markers=markers, decimal_comma=True
        )
        self.assertEqual(mask, [True, True, False, True])

    def test_unknown_target_raises(self):
        with self.assertRaises(ValueError):
            build_design_matrix(["a"], [["1"]], target="nope")


class EncodeTargetTest(unittest.TestCase):
    def test_classification_labels_are_sorted_and_encoded(self):
        y, names = encode_target(["cat", "dog", "cat", "bird"], "classification")
        self.assertEqual(names, ["bird", "cat", "dog"])
        np.testing.assert_array_equal(y, [1, 2, 1, 0])

    def test_regression_returns_floats(self):
        y, names = encode_target(["1,5", "2,5"], "regression", decimal_comma=True)
        np.testing.assert_allclose(y, [1.5, 2.5])
        self.assertEqual(names, [])

    def test_missing_target_raises(self):
        with self.assertRaises(ValueError):
            encode_target(["1", "?"], "regression")


class NoLeakageTest(unittest.TestCase):
    """防泄露：所有统计量只能来自训练集。"""

    def test_imputation_uses_training_mean_only(self):
        X_train = np.array([[0.0], [10.0]])          # 训练均值 = 5
        X_test = np.array([[np.nan], [1000.0]])      # 测试集有极端值
        filled_train, filled_test, means = impute_with_train_statistics(X_train, X_test)

        self.assertAlmostEqual(means[0], 5.0)
        # 若用全局均值 (0+10+1000)/3 = 336.7 填充，这里就会失败
        self.assertAlmostEqual(filled_test[0, 0], 5.0)
        np.testing.assert_allclose(filled_train[:, 0], [0.0, 10.0])

    def test_imputation_is_independent_of_test_set(self):
        X_train = np.array([[0.0], [10.0]])
        train_a, _, means_a = impute_with_train_statistics(X_train, np.array([[np.nan]]))
        train_b, _, means_b = impute_with_train_statistics(
            X_train, np.array([[999.0], [np.nan]])
        )
        np.testing.assert_allclose(means_a, means_b)
        np.testing.assert_allclose(train_a, train_b)

    def test_imputation_of_all_missing_column_is_zero(self):
        X_train = np.array([[np.nan, 1.0], [np.nan, 3.0]])
        filled, means = impute_with_train_statistics(X_train)
        self.assertAlmostEqual(means[0], 0.0)
        self.assertTrue(np.isfinite(filled).all())

    def test_standardize_uses_training_statistics_only(self):
        X_train = np.array([[0.0], [2.0]])       # 均值 1，标准差 1
        X_test = np.array([[1000.0]])
        train, test, mean, scale = standardize(X_train, X_test)
        self.assertAlmostEqual(mean[0], 1.0)
        self.assertAlmostEqual(scale[0], 1.0)
        np.testing.assert_allclose(train[:, 0], [-1.0, 1.0])
        self.assertAlmostEqual(test[0, 0], 999.0)


class MissingReportTest(unittest.TestCase):
    def test_counts_per_column(self):
        header = ["a", "b"]
        rows = [["1", "?"], ["?", "2"], ["3", "?"]]
        report = missing_report(header, rows, ["", "?"])
        self.assertEqual(report, {"a": 1, "b": 2})


class DuplicateRowsTest(unittest.TestCase):
    def test_counts_exact_duplicates(self):
        rows = [["1", "a"], ["2", "b"], ["1", "a"], ["1", "a"]]
        self.assertEqual(duplicate_row_count(rows), 2)

    def test_whitespace_only_differences_count_as_duplicates(self):
        """去空白后再比较，避免 ' a ' 与 'a' 被当成两行。"""
        rows = [["1", "a"], [" 1 ", " a "]]
        self.assertEqual(duplicate_row_count(rows), 1)

    def test_deduplicate_keeps_first_occurrence_order(self):
        rows = [["1", "a"], ["2", "b"], ["1", "a"]]
        unique, removed = deduplicate_rows(rows)
        self.assertEqual(removed, 1)
        self.assertEqual(unique, [["1", "a"], ["2", "b"]])

    def test_rows_differing_only_in_target_are_not_duplicates(self):
        """特征相同但标签不同属于冲突样本，不能当重复行删掉。"""
        rows = [["1", "a", "yes"], ["1", "a", "no"]]
        self.assertEqual(duplicate_row_count(rows), 0)

    def test_no_duplicates_returns_zero(self):
        rows = [["1"], ["2"], ["3"]]
        self.assertEqual(duplicate_row_count(rows), 0)
        unique, removed = deduplicate_rows(rows)
        self.assertEqual(removed, 0)
        self.assertEqual(unique, rows)


if __name__ == "__main__":
    unittest.main()
