"""噪声注入模块 - 用于鲁棒性测试"""
import numpy as np
import pandas as pd


def add_gaussian_noise(X, noise_level=0.01):
    """添加高斯噪声"""
    X_noisy = X.copy()
    for col in X_noisy.columns:
        std = X_noisy[col].std()
        noise = np.random.normal(0, noise_level * std, size=len(X_noisy))
        X_noisy[col] = X_noisy[col] + noise
    return X_noisy


def add_label_noise(y, noise_level=0.05, num_classes=None):
    """添加标签噪声（随机翻转一定比例的标签）"""
    y_noisy = y.copy()
    unique_classes = list(set(y))
    n_samples = len(y_noisy)
    n_noisy = int(n_samples * noise_level)
    indices = np.random.choice(n_samples, n_noisy, replace=False)
    for idx in indices:
        original = y_noisy.iloc[idx]
        other_classes = [c for c in unique_classes if c != original]
        y_noisy.iloc[idx] = np.random.choice(other_classes)
    return y_noisy


def add_missing_values(X, missing_rate=0.05):
    """随机将值置为 NaN 模拟额外缺失值"""
    X_missing = X.copy()
    mask = np.random.random(X_missing.shape) < missing_rate
    X_missing = X_missing.mask(pd.DataFrame(mask, columns=X_missing.columns, index=X_missing.index))
    return X_missing


def generate_robustness_test_data(X_test, y_test, noise_configs):
    """生成不同噪声配置下的测试数据

    noise_configs: list of dict, e.g.:
        [{"type": "gaussian", "level": 0.01}, ...]
    返回: dict {label: (X_noisy, y_noisy)}
    """
    datasets = {}
    for cfg in noise_configs:
        label = f"{cfg['type']}_{cfg['level']}"
        X_n = X_test.copy()
        y_n = y_test.copy()

        if cfg["type"] == "gaussian":
            X_n = add_gaussian_noise(X_n, cfg["level"])
        elif cfg["type"] == "label":
            y_n = add_label_noise(y_n, cfg["level"])
        elif cfg["type"] == "missing":
            X_n = add_missing_values(X_n, cfg["level"])
            from .preprocessor import impute_missing_values
            X_n = impute_missing_values(X_n)

        datasets[label] = (X_n, y_n)

    return datasets
