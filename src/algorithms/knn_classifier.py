"""KNN 分类器 - 课内算法"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from .base import BaseClassifier, AlgorithmFactory


class KNNClassifier(BaseClassifier):
    def __init__(self, n_neighbors=5, **kwargs):
        super().__init__("KNN")
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def build(**kwargs):
    return KNNClassifier(**kwargs)


AlgorithmFactory.register("knn", build)
