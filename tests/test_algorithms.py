import unittest

import numpy as np

from backend.algorithms import (
    DecisionTree,
    GradientBoosting,
    GaussianNaiveBayes,
    KMeans,
    KNN,
    LinearRegression,
    LogisticRegression,
    PCA,
    RandomForest,
    SVM,
)


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

    def test_supervised_five_classifiers(self):
        for model in [
            LogisticRegression(n_iters=500),
            KNN(k=3),
            DecisionTree(max_depth=4),
            SVM(n_iters=500),
        ]:
            model.fit(self.X, self.y)
            predictions = model.predict(self.X)
            self.assertEqual(predictions.shape, self.y.shape)
            self.assertEqual(model.predict_proba(self.X).shape, (80, 2))

    def test_linear_regression(self):
        model = LinearRegression().fit(self.X, self.regression_y)
        predictions = model.predict(self.X)
        self.assertEqual(predictions.shape, self.regression_y.shape)
        self.assertLess(float(np.mean((predictions - self.regression_y) ** 2)), 1e-6)

    def test_classifiers_separable_accuracy(self):
        X = np.vstack([np.random.default_rng(0).normal(-2, 0.5, (30, 2)), np.random.default_rng(1).normal(2, 0.5, (30, 2))])
        y = np.array(["a"] * 30 + ["b"] * 30)
        for model in [KNN(k=3), DecisionTree(max_depth=3), LogisticRegression(), SVM()]:
            accuracy = float(np.mean(model.fit(X, y).predict(X) == y))
            self.assertGreater(accuracy, 0.95, msg=f"{type(model).__name__} failed on separable data")

    def test_cross_validation_service(self):
        from backend.services.train_service import run_cross_validation

        result = run_cross_validation("knn", self.X, self.y, n_splits=4, k=3)
        self.assertEqual(len(result["folds"]), 4)
        self.assertIn("accuracy", result["summary"])
        self.assertIn("mean", result["summary"]["accuracy"])


if __name__ == "__main__":
    unittest.main()
