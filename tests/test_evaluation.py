# -*- coding: utf-8 -*-
"""指标计算测试。

聚类指标是本次新加的（轮廓系数、Calinski-Harabasz），所以用「分离良好」
与「完全重叠」两组对照数据来验证它们的方向性，而不只是验证能跑通。
"""
from __future__ import annotations

import unittest

import numpy as np

from backend.services.evaluation import (
    calinski_harabasz_score,
    classification_metrics,
    clustering_metrics,
    regression_metrics,
    silhouette_score,
)


class ClassificationMetricsTest(unittest.TestCase):
    def test_confusion_matrix_and_scores(self):
        result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(result["labels"], [0, 1])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["precision_macro"], (1.0 + 2 / 3) / 2)
        self.assertAlmostEqual(result["recall_macro"], (0.5 + 1.0) / 2)

    def test_perfect_prediction(self):
        result = classification_metrics([0, 1, 2], [0, 1, 2])
        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["f1_macro"], 1.0)
        self.assertAlmostEqual(result["confusion_matrix"][0][0], 1)


class RegressionMetricsTest(unittest.TestCase):
    def test_perfect_prediction(self):
        result = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(result["mae"], 0.0)
        self.assertAlmostEqual(result["rmse"], 0.0)
        self.assertAlmostEqual(result["r2"], 1.0)

    def test_constant_prediction_at_the_mean_has_zero_r2(self):
        result = regression_metrics([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
        self.assertAlmostEqual(result["mae"], 2 / 3)
        self.assertAlmostEqual(result["mse"], 2 / 3)
        self.assertAlmostEqual(result["r2"], 0.0)

    def test_known_values(self):
        result = regression_metrics([1.0, 2.0], [2.0, 4.0])
        self.assertAlmostEqual(result["mae"], 1.5)
        self.assertAlmostEqual(result["mse"], 2.5)
        self.assertAlmostEqual(result["rmse"], np.sqrt(2.5))


class ClusteringMetricsTest(unittest.TestCase):
    @staticmethod
    def _blobs(separation: float, seed: int = 0):
        rng = np.random.default_rng(seed)
        left = rng.normal(0.0, 0.1, (40, 2))
        right = rng.normal(separation, 0.1, (40, 2))
        X = np.vstack([left, right])
        labels = np.array([0] * 40 + [1] * 40)
        return X, labels

    def test_separated_clusters_score_high(self):
        X, labels = self._blobs(separation=5.0)
        self.assertGreater(silhouette_score(X, labels), 0.9)
        self.assertGreater(calinski_harabasz_score(X, labels), 100.0)

    def test_overlapping_clusters_score_lower_than_separated(self):
        far_X, far_labels = self._blobs(separation=5.0)
        near_X, near_labels = self._blobs(separation=0.05)
        self.assertLess(
            silhouette_score(near_X, near_labels),
            silhouette_score(far_X, far_labels),
        )

    def test_chunking_does_not_change_the_result(self):
        """分块计算必须与整块计算等价，否则大样本上的指标不可信。"""
        X, labels = self._blobs(separation=3.0)
        self.assertAlmostEqual(
            silhouette_score(X, labels, chunk_size=256),
            silhouette_score(X, labels, chunk_size=7),
        )

    def test_single_cluster_raises(self):
        with self.assertRaises(ValueError):
            silhouette_score(np.zeros((5, 2)), np.zeros(5, dtype=int))
        with self.assertRaises(ValueError):
            calinski_harabasz_score(np.zeros((5, 2)), np.zeros(5, dtype=int))

    def test_clustering_metrics_reports_sizes_and_sampling(self):
        X, labels = self._blobs(separation=4.0)
        result = clustering_metrics(X, labels)
        self.assertEqual(result["n_clusters"], 2)
        self.assertEqual(result["cluster_sizes"], {"0": 40, "1": 40})
        self.assertFalse(result["silhouette_sampled"])
        self.assertAlmostEqual(result["silhouette_sample_size"], 80)

        sampled = clustering_metrics(X, labels, max_samples=20)
        self.assertTrue(sampled["silhouette_sampled"])
        self.assertLessEqual(sampled["silhouette_sample_size"], 20)
        self.assertEqual(sampled["n_clusters"], 2)

    def test_clustering_metrics_handles_single_cluster_gracefully(self):
        result = clustering_metrics(np.zeros((5, 2)), np.zeros(5, dtype=int))
        self.assertEqual(result["n_clusters"], 1)
        self.assertIsNone(result["silhouette"])
        self.assertIsNone(result["calinski_harabasz"])


if __name__ == "__main__":
    unittest.main()
