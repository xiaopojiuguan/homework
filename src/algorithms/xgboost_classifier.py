"""XGBoost 分类器 - 课外算法（课堂上没讲过）"""
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from .base import BaseClassifier, AlgorithmFactory


class XGBoostClassifier(BaseClassifier):
    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.1, **kwargs):
        super().__init__("XGBoost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.label_encoder = None
        self.classes_ = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_train_enc = self.label_encoder.fit_transform(y_train)
        self.classes_ = self.label_encoder.classes_

        if X_val is not None and y_val is not None:
            y_val_enc = self.label_encoder.transform(y_val)
            self.model = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
                n_jobs=-1,
                eval_metric="mlogloss",
                early_stopping_rounds=20,
            )
            self.model.fit(
                X_train, y_train_enc,
                eval_set=[(X_train, y_train_enc), (X_val, y_val_enc)],
                verbose=False,
            )
            results = self.model.evals_result()
            self.train_losses = results["validation_0"]["mlogloss"]
            self.val_losses = results["validation_1"]["mlogloss"]
        else:
            self.model = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
                n_jobs=-1,
                eval_metric="mlogloss",
            )
            self.model.fit(X_train, y_train_enc, verbose=False)
            results = self.model.evals_result()
            self.train_losses = results["validation_0"]["mlogloss"]
            self.val_losses = []

        self.is_trained = True
        return self

    def predict(self, X):
        y_enc = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_enc.astype(int))

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def build(**kwargs):
    return XGBoostClassifier(**kwargs)


AlgorithmFactory.register("xgboost", build)
