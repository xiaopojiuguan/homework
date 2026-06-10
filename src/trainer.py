"""训练调度模块 - 统一训练所有算法"""
import os
import json
import time
import joblib
import numpy as np
from .algorithms.base import AlgorithmFactory

# 导入所有算法以注册
from .algorithms import logistic_regression
from .algorithms import random_forest
from .algorithms import xgboost_classifier
from .algorithms import svm_classifier
from .algorithms import knn_classifier
from .algorithms import ann_classifier

MODEL_DIR = "outputs/models"
RESULTS_DIR = "outputs/results"


def train_all_algorithms(X_train, y_train, X_val, y_val, algos=None):
    """训练所有已注册的算法"""
    if algos is None:
        algos = AlgorithmFactory.list_algorithms()

    results = {}
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for algo_name in algos:
        print(f"\n{'='*50}")
        print(f"Training: {algo_name}")
        print(f"{'='*50}")

        clf = AlgorithmFactory.create(algo_name)
        t_start = time.time()
        clf.fit(X_train, y_train, X_val, y_val)
        train_time = time.time() - t_start

        results[algo_name] = {
            "name": clf.name,
            "train_time_seconds": train_time,
        }

        # 保存模型
        model_path = os.path.join(MODEL_DIR, f"{algo_name}.pkl")
        joblib.dump(clf, model_path)

        print(f"  Training completed in {train_time:.2f}s")
        print(f"  Model saved to {model_path}")

    return results


def load_trained_model(algo_name):
    """加载已训练的模型"""
    model_path = os.path.join(MODEL_DIR, f"{algo_name}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def train_single_algorithm(algo_name, X_train, y_train, X_val, y_val):
    """训练单个算法"""
    clf = AlgorithmFactory.create(algo_name)
    t_start = time.time()
    clf.fit(X_train, y_train, X_val, y_val)
    train_time = time.time() - t_start

    model_path = os.path.join(MODEL_DIR, f"{algo_name}.pkl")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, model_path)

    result = {
        "name": clf.name,
        "train_time_seconds": train_time,
    }
    return clf, result
