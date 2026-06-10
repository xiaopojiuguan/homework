"""随机森林分类器 - 课内算法"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base import BaseClassifier, AlgorithmFactory


class RandomForestClassifier_(BaseClassifier):
    def __init__(self, n_estimators=200, max_depth=15, **kwargs):
        super().__init__("Random Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True
        # RF 不是迭代训练型，不产生 loss 曲线
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def build(**kwargs):
    return RandomForestClassifier_(**kwargs)


AlgorithmFactory.register("random_forest", build)
