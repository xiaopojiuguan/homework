"""逻辑回归分类器 - 课内算法"""
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from .base import BaseClassifier, AlgorithmFactory


class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self, max_iter=5000, C=1.0, **kwargs):
        super().__init__("Logistic Regression")
        self.max_iter = max_iter
        self.C = C
        self.scaler = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)

        self.model = LogisticRegression(
            max_iter=self.max_iter,
            C=self.C,
            solver="lbfgs",
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_train_s, y_train)

        self.train_losses = []
        self.val_losses = []

        # 使用 warm_start 追踪 loss 曲线（仅取 50 个点减少计算量）
        n_points = 50
        iter_step = max(1, self.max_iter // n_points)
        m = LogisticRegression(
            max_iter=1, C=self.C, solver="lbfgs",
            random_state=42, warm_start=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(1, self.max_iter + 1, iter_step):
                m.max_iter = iter_step
                m.fit(X_train_s, y_train)
                yp_train = m.predict_proba(X_train_s)
                self.train_losses.append(log_loss(y_train, yp_train))
                if X_val is not None and y_val is not None:
                    X_val_s = self.scaler.transform(X_val)
                    yp_val = m.predict_proba(X_val_s)
                    self.val_losses.append(log_loss(y_val, yp_val))

        self.is_trained = True
        return self

    def predict(self, X):
        X_s = self.scaler.transform(X)
        return self.model.predict(X_s)

    def predict_proba(self, X):
        X_s = self.scaler.transform(X)
        return self.model.predict_proba(X_s)


def build(**kwargs):
    return LogisticRegressionClassifier(**kwargs)


AlgorithmFactory.register("logistic_regression", build)
