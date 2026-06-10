"""数据预处理模块 - 标签清洗、缺失值处理、特征工程"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


LABEL_CORRECTIONS = {
    "D3RMAS0N": "DERMASON",
    "S3K3R": "SEKER",
    "H0R0Z": "HOROZ",
    "B0MBAY": "BOMBAY",
}


def clean_labels(y):
    """清洗标签：去空格 -> 转大写 -> 纠正拼写错误"""
    y = y.astype(str).str.strip().str.upper()
    y = y.replace(LABEL_CORRECTIONS)
    return y


def impute_missing_values(X, y=None):
    """缺失值处理：按类别分组用中位数填充（如不提供y则全局中位数）"""
    cols_with_nulls = X.columns[X.isnull().any()].tolist()

    if y is not None and len(cols_with_nulls) > 0:
        df = X.copy()
        df["_Class_"] = y.values
        for col in cols_with_nulls:
            medians = df.groupby("_Class_")[col].transform("median")
            df[col] = df[col].fillna(medians)
            df[col] = df[col].fillna(df[col].median())
        X = df.drop(columns=["_Class_"])
    elif len(cols_with_nulls) > 0:
        imputer = SimpleImputer(strategy="median")
        X[cols_with_nulls] = imputer.fit_transform(X[cols_with_nulls])

    return X


def scale_features(X_train, X_test, X_val):
    """StandardScaler 标准化"""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns, index=X_val.index
    )
    return X_train_scaled, X_test_scaled, X_val_scaled, scaler


def clean_numerical_columns(X):
    """清理数值列：去除单位后缀、将字符串转为数值"""
    for col in X.columns:
        if X[col].dtype == object:
            # 去除常见单位后缀 " cm", " mm", " px" 等
            X[col] = X[col].astype(str).str.replace(r"\s*(cm|mm|px|in)$", "", regex=True)
            X[col] = X[col].str.strip()
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def preprocess_pipeline(train_df, test_df, val_df):
    """完整预处理流水线"""
    from .data_loader import get_feature_target

    # 分离特征和标签
    X_train, y_train_raw = get_feature_target(train_df)
    X_test, y_test_raw = get_feature_target(test_df)
    X_val, y_val_raw = get_feature_target(val_df)

    # 清理数值列中的非数值内容（如单位后缀）
    X_train = clean_numerical_columns(X_train)
    X_test = clean_numerical_columns(X_test)
    X_val = clean_numerical_columns(X_val)

    # 清洗标签
    y_train = clean_labels(y_train_raw)
    y_test = clean_labels(y_test_raw)
    y_val = clean_labels(y_val_raw)

    # 缺失值填充（按类别分组）
    X_train = impute_missing_values(X_train, y_train)
    X_test = impute_missing_values(X_test, y_test)
    X_val = impute_missing_values(X_val, y_val)

    # 标准化
    X_train_scaled, X_test_scaled, X_val_scaled, scaler = scale_features(
        X_train, X_test, X_val
    )

    processed = {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "X_val": X_val_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "y_val": y_val,
        "scaler": scaler,
        "classes": sorted(y_train.unique().tolist()),
    }
    return processed
