"""Self-contained machine learning algorithms used by the platform."""

from .decision_tree import DecisionTree
from .gradient_boosting import GradientBoosting
from .kmeans import KMeans
from .knn import KNN
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .naive_bayes import GaussianNaiveBayes
from .pca import PCA
from .random_forest import RandomForest
from .svm import SVM

__all__ = [
    "KMeans",
    "RandomForest",
    "GaussianNaiveBayes",
    "PCA",
    "GradientBoosting",
    "LinearRegression",
    "LogisticRegression",
    "KNN",
    "DecisionTree",
    "SVM",
]
