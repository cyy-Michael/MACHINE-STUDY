# -*- coding: utf-8 -*-
"""端到端编排测试（需要本地数据集）。"""
from __future__ import annotations

import json
import unittest

import numpy as np

from backend.services.experiment import (
    ExperimentError,
    algorithm_task_type,
    json_safe,
    run_dataset_experiment,
)
from tests.support import requires_data


class AlgorithmTaskTypeTest(unittest.TestCase):
    def test_known_algorithms(self):
        self.assertEqual(algorithm_task_type("knn"), "classification")
        self.assertEqual(algorithm_task_type("linear_regression"), "regression")
        self.assertEqual(algorithm_task_type("kmeans"), "clustering")

    def test_unknown_algorithm_raises(self):
        with self.assertRaises(ExperimentError):
            algorithm_task_type("no_such_algorithm")


class JsonSafeTest(unittest.TestCase):
    def test_non_finite_floats_become_none(self):
        """inf/nan 不是合法 JSON，必须转成 null。"""
        payload = json_safe(
            {"a": float("inf"), "b": float("-inf"), "c": float("nan"), "d": 1.5}
        )
        self.assertIsNone(payload["a"])
        self.assertIsNone(payload["b"])
        self.assertIsNone(payload["c"])
        self.assertEqual(payload["d"], 1.5)

    def test_numpy_scalars_become_python_types(self):
        payload = json_safe({"x": np.float64(1.5), "y": np.int64(3), "z": np.array([1, 2])})
        self.assertEqual(payload, {"x": 1.5, "y": 3, "z": [1, 2]})
        json.dumps(payload, allow_nan=False)

    def test_bool_is_not_turned_into_number(self):
        self.assertIs(json_safe({"flag": True})["flag"], True)


class ValidationTest(unittest.TestCase):
    def test_bad_test_size(self):
        with self.assertRaises(ExperimentError):
            run_dataset_experiment("iris", "knn", test_size=1.5)

    def test_task_mismatch(self):
        """分类数据集配纯回归算法应当被拦住。"""
        with self.assertRaises(ExperimentError):
            run_dataset_experiment("iris", "linear_regression")

    def test_unknown_algorithm(self):
        with self.assertRaises(ExperimentError):
            run_dataset_experiment("iris", "no_such_algorithm")

    def test_task_parameter_rejected_for_plain_algorithm(self):
        with self.assertRaises(ExperimentError):
            run_dataset_experiment("iris", "knn", {"task": "classification"})


@requires_data
class ClassificationExperimentTest(unittest.TestCase):
    def test_iris_knn_end_to_end(self):
        result = run_dataset_experiment(
            "iris", "knn", {"k": 5}, random_state=42
        )
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["dataset"], "iris")
        self.assertEqual(result["task_type"], "classification")
        self.assertGreater(result["metrics"]["accuracy"], 0.8)
        self.assertEqual(result["split"]["n_train"] + result["split"]["n_test"], 150)
        self.assertIsNone(result["error"])

    def test_result_is_strictly_json_serializable(self):
        """接口层要直接返回它，所以必须能被 allow_nan=False 序列化。"""
        result = run_dataset_experiment("iris", "decision_tree", {"max_depth": 3})
        json.dumps(result, allow_nan=False)

    def test_model_object_is_not_returned(self):
        """P0 缺陷 #8：模型实例不可序列化，绝不能出现在结果里。"""
        result = run_dataset_experiment("iris", "knn", {"k": 3})
        self.assertNotIn("model", result)
        json.dumps(result, allow_nan=False)

    def test_train_and_predict_times_are_separate(self):
        """P0 缺陷 #7：训练与预测必须分别计时。"""
        result = run_dataset_experiment("iris", "knn", {"k": 3})
        self.assertIn("train_time_seconds", result)
        self.assertIn("predict_time_seconds", result)
        self.assertNotIn("prediction_time_seconds", result)
        self.assertGreaterEqual(result["train_time_seconds"], 0.0)
        self.assertGreaterEqual(result["predict_time_seconds"], 0.0)

    def test_metrics_include_required_fields(self):
        result = run_dataset_experiment("iris", "knn", {"k": 3})
        metrics = result["metrics"]
        for field in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                      "confusion_matrix", "labels", "label_names"):
            with self.subTest(field=field):
                self.assertIn(field, metrics)

    def test_confusion_matrix_is_square_and_consistent(self):
        result = run_dataset_experiment("iris", "knn", {"k": 3})
        matrix = np.array(result["metrics"]["confusion_matrix"])
        self.assertEqual(matrix.shape[0], matrix.shape[1])
        self.assertEqual(matrix.sum(), result["split"]["n_test"])
        self.assertEqual(matrix.shape[0], len(result["metrics"]["label_names"]))

    def test_predictions_are_truncated_and_flagged(self):
        result = run_dataset_experiment("iris", "knn", {"k": 3}, max_predictions=5)
        self.assertEqual(len(result["predictions"]), 5)
        self.assertTrue(result["predictions_truncated"])
        self.assertEqual(result["predictions_total"], result["split"]["n_test"])

    def test_visualization_has_scatter_for_classification(self):
        result = run_dataset_experiment("iris", "knn", {"k": 3})
        self.assertIn("scatter_2d", result["visualization"])
        self.assertTrue(result["visualization"]["scatter_2d"])

    def test_same_seed_gives_same_result(self):
        first = run_dataset_experiment("iris", "knn", {"k": 5}, random_state=7)
        second = run_dataset_experiment("iris", "knn", {"k": 5}, random_state=7)
        self.assertAlmostEqual(first["metrics"]["accuracy"], second["metrics"]["accuracy"])

    def test_standardization_flag_is_recorded(self):
        result = run_dataset_experiment("iris", "knn", {"k": 3}, standardize_features=False)
        self.assertFalse(result["preprocessing"]["standardized"])
        self.assertIsNone(result["preprocessing"]["scaler"])


@requires_data
class RegressionExperimentTest(unittest.TestCase):
    def test_bike_day_linear_regression(self):
        result = run_dataset_experiment("bike_day", "linear_regression")
        self.assertEqual(result["task_type"], "regression")
        for field in ("mae", "mse", "rmse", "r2"):
            with self.subTest(field=field):
                self.assertIn(field, result["metrics"])
        # casual + registered 已剔除，所以 R2 必须是正常水平而不是 1.0
        self.assertLess(result["metrics"]["r2"], 0.99)
        self.assertGreater(result["metrics"]["r2"], 0.5)

    def test_visualization_has_true_vs_pred_for_regression(self):
        result = run_dataset_experiment("concrete", "linear_regression")
        self.assertIn("true_vs_pred", result["visualization"])
        self.assertTrue(result["visualization"]["true_vs_pred"])


@requires_data
class ClusteringExperimentTest(unittest.TestCase):
    def test_kmeans_on_iris_reports_clustering_metrics(self):
        result = run_dataset_experiment("iris", "kmeans", {"n_clusters": 3})
        self.assertEqual(result["task_type"], "clustering")
        self.assertEqual(result["metrics"]["n_clusters"], 3)
        self.assertIn("silhouette", result["metrics"])
        self.assertIn("calinski_harabasz", result["metrics"])
        self.assertGreater(result["metrics"]["silhouette"], 0.3)

    def test_clustering_allowed_on_regression_dataset(self):
        """聚类不依赖标签，回归数据集也应该能跑。"""
        result = run_dataset_experiment("bike_day", "kmeans", {"n_clusters": 2})
        self.assertEqual(result["task_type"], "clustering")


if __name__ == "__main__":
    unittest.main()
