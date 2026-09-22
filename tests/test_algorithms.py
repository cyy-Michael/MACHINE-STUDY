import unittest

import numpy as np

from backend.algorithms import GradientBoosting, GaussianNaiveBayes, KMeans, PCA, RandomForest


class AlgorithmsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.X = rng.normal(size=(80, 3))
        self.y = np.where(self.X[:, 0] + self.X[:, 1] > 0, "yes", "no")
        self.regression_y = 1.5 * self.X[:, 0] - 0.5 * self.X[:, 1]

    def test_kmeans(self):
        model = KMeans(n_clusters=2).fit(self.X)
        self.assertEqual(model.predict(self.X).shape, (80,))
        self.assertEqual(model.transform(self.X).shape, (80, 2))

    def test_pca(self):
        model = PCA(n_components=2).fit(self.X)
        self.assertEqual(model.transform(self.X).shape, (80, 2))
        self.assertEqual(model.inverse_transform(model.transform(self.X)).shape, self.X.shape)

    def test_classifiers(self):
        for model in [
            GaussianNaiveBayes(),
            RandomForest(n_estimators=8, max_depth=4),
            GradientBoosting(task="classification", n_estimators=8),
        ]:
            model.fit(self.X, self.y)
            predictions = model.predict(self.X)
            self.assertEqual(predictions.shape, self.y.shape)
            self.assertEqual(model.predict_proba(self.X).shape, (80, 2))

    def test_regressors(self):
        for model in [
            RandomForest(task="regression", n_estimators=8, max_depth=4),
            GradientBoosting(task="regression", n_estimators=8),
        ]:
            model.fit(self.X, self.regression_y)
            self.assertEqual(model.predict(self.X).shape, self.regression_y.shape)


if __name__ == "__main__":
    unittest.main()
