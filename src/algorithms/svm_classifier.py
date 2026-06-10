"""SVM 分类器 - 课内/课外算法"""
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from .base import BaseClassifier, AlgorithmFactory


class SVMClassifier(BaseClassifier):
    def __init__(self, kernel="rbf", C=1.0, **kwargs):
        super().__init__("SVM")
        self.kernel = kernel
        self.C = C
        self.scaler = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)

        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            probability=True,
            random_state=42,
        )
        self.model.fit(X_train_s, y_train)
        self.is_trained = True
        return self

    def predict(self, X):
        X_s = self.scaler.transform(X)
        return self.model.predict(X_s)

    def predict_proba(self, X):
        X_s = self.scaler.transform(X)
        return self.model.predict_proba(X_s)


def build(**kwargs):
    return SVMClassifier(**kwargs)


AlgorithmFactory.register("svm", build)
