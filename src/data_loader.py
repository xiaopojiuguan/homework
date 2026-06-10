"""数据加载模块 - 加载 Dry Bean Dataset CSV 文件"""
import os
import pandas as pd
import numpy as np


def _get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data(data_dir=None):
    """加载训练集、测试集、验证集"""
    if data_dir is None:
        data_dir = os.path.join(_get_project_root(), "数据")
    train_path = os.path.join(data_dir, "Dry_Bean_Dataset_Dirty_train.csv")
    test_path = os.path.join(data_dir, "Dry_Bean_Dataset_Dirty_test.csv")
    val_path = os.path.join(data_dir, "Dry_Bean_Dataset_Dirty_val.csv")

    train_df = pd.read_csv(train_path, na_values=["?", "NA", "nan", ""])
    test_df = pd.read_csv(test_path, na_values=["?", "NA", "nan", ""])
    val_df = pd.read_csv(val_path, na_values=["?", "NA", "nan", ""])

    # 去除 BOM 字符（如果列名中有）
    train_df.columns = train_df.columns.str.replace("﻿", "")
    test_df.columns = test_df.columns.str.replace("﻿", "")
    val_df.columns = val_df.columns.str.replace("﻿", "")

    return train_df, test_df, val_df


def get_feature_target(df):
    """分离特征和目标变量"""
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def get_data_summary(train_df, test_df, val_df):
    """获取数据摘要信息"""
    summary = {
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "val_shape": val_df.shape,
        "features": list(train_df.drop(columns=["Class"]).columns),
        "num_features": train_df.shape[1] - 1,
        "null_counts": {
            "train": train_df.isnull().sum().to_dict(),
            "test": test_df.isnull().sum().to_dict(),
            "val": val_df.isnull().sum().to_dict(),
        },
        "class_distribution": {
            "train": train_df["Class"].value_counts().to_dict(),
            "test": test_df["Class"].value_counts().to_dict(),
            "val": val_df["Class"].value_counts().to_dict(),
        },
        "unique_classes_raw": sorted(train_df["Class"].unique().tolist()),
        "num_classes_raw": train_df["Class"].nunique(),
    }
    return summary
