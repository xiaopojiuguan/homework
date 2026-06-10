"""
Dry Bean Dataset - 机器学习全流程工程项目
============================================
CLI 统一入口

用法:
    python main.py analyze         # 数据分析 + 生成图表
    python main.py preprocess      # 数据预处理
    python main.py train           # 训练所有算法
    python main.py evaluate        # 多维度评估分析
    python main.py full-pipeline   # 一键运行全流程
"""
import sys
import os
import argparse
import json
import warnings

warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# 确保工作目录为项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

from src.data_loader import load_data, get_feature_target, get_data_summary
from src.preprocessor import preprocess_pipeline, clean_labels
from src.trainer import train_all_algorithms, train_single_algorithm
from src.evaluator import run_full_evaluation
from src import visualizer


def cmd_analyze():
    """数据分析命令"""
    print("=" * 60)
    print("STEP 1: DATA ANALYSIS")
    print("=" * 60)

    train_df, test_df, val_df = load_data()

    # 数据摘要
    summary = get_data_summary(train_df, test_df, val_df)
    print(f"\nTrain shape: {summary['train_shape']}")
    print(f"Test shape:  {summary['test_shape']}")
    print(f"Val shape:   {summary['val_shape']}")
    print(f"\nNumber of raw classes: {summary['num_classes_raw']}")
    print(f"Raw class names: {summary['unique_classes_raw']}")
    print(f"\nMissing values (train):")
    for k, v in summary["null_counts"]["train"].items():
        if v > 0:
            print(f"  {k}: {v}")

    # 数据污染分析
    print("\n--- Data Contamination Analysis ---")
    y_raw = train_df["Class"]
    print(f"Raw unique labels in Class: {y_raw.nunique()}")
    print(f"Raw labels: {sorted(y_raw.unique())}")

    # 分析标签问题
    import pandas as pd
    y_cleaned = y_raw.astype(str).str.strip().str.upper()
    correct_set = {"DERMASON", "SIRA", "SEKER", "HOROZ", "CALI", "BARBUNYA", "BOMBAY"}
    abnormal = [v for v in y_cleaned.unique() if v not in correct_set]
    print(f"\nAbnormal labels detected (should map to correct classes):")
    for a in abnormal:
        count = (y_cleaned == a).sum()
        print(f"  '{a}': {count} samples")

    # 生成数据分布图
    print("\nGenerating data analysis figures...")
    out_dir = "outputs/figures/data_analysis"

    visualizer.plot_class_distribution(
        train_df["Class"],
        "Raw Class Distribution (Training Set)",
        os.path.join(out_dir, "class_distribution.png")
    )

    visualizer.plot_feature_distributions(
        train_df.head(1000), "Class",
        out_dir
    )

    visualizer.plot_correlation_heatmap(
        train_df.drop(columns=["Class"]).head(3000),
        os.path.join(out_dir, "correlation_heatmap.png")
    )

    visualizer.plot_null_heatmap(
        train_df,
        "Missing Values Heatmap (Training Set)",
        os.path.join(out_dir, "null_heatmap.png")
    )

    # 生成文本分析报告
    report = f"""
Data Analysis Report
====================
Dataset: Dry Bean Dataset (Dirty Version)
Source: UCI Machine Learning Repository

1. Data Overview:
   - Training samples: {summary['train_shape'][0]}
   - Test samples: {summary['test_shape'][0]}
   - Validation samples: {summary['val_shape'][0]}
   - Number of features: {summary['num_features']}
   - Feature types: All numerical

2. Data Contamination Issues:
   a) Missing Values: Perimeter ({summary['null_counts']['train']['Perimeter']} in train,
      {summary['null_counts']['test']['Perimeter']} in test, {summary['null_counts']['val']['Perimeter']} in val)
      and Solidity ({summary['null_counts']['train']['Solidity']} in train,
      {summary['null_counts']['test']['Solidity']} in test, {summary['null_counts']['val']['Solidity']} in val)
   b) Label Noise: {len(abnormal)} abnormal label variants detected, including
      typos (D3RMAS0N), case errors (dermason), and trailing spaces.
   c) True classes: 7 bean types - DERMASON, SIRA, SEKER, HOROZ, CALI, BARBUNYA, BOMBAY

3. Class Imbalance:
   - DERMASON is the largest class, BOMBAY is the smallest
   - Moderate imbalance present, acceptable for multi-class classification
"""
    with open("outputs/results/data_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("Data analysis complete. Figures saved to outputs/figures/data_analysis/")


def cmd_preprocess():
    """数据预处理命令"""
    print("=" * 60)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 60)

    train_df, test_df, val_df = load_data()
    processed = preprocess_pipeline(train_df, test_df, val_df)

    print(f"\nPreprocessing complete:")
    print(f"  X_train shape: {processed['X_train'].shape}")
    print(f"  X_test shape:  {processed['X_test'].shape}")
    print(f"  X_val shape:   {processed['X_val'].shape}")
    print(f"  Classes: {processed['classes']}")
    print(f"  Class counts in train: {processed['y_train'].value_counts().to_dict()}")

    return processed


def cmd_train():
    """训练所有算法"""
    print("=" * 60)
    print("STEP 3: MODEL TRAINING")
    print("=" * 60)

    train_df, test_df, val_df = load_data()
    processed = preprocess_pipeline(train_df, test_df, val_df)

    algo_names = ["knn", "logistic_regression", "svm", "ann", "xgboost"]
    results = train_all_algorithms(
        processed["X_train"], processed["y_train"],
        processed["X_val"], processed["y_val"],
        algos=algo_names,
    )

    print("\nTraining Summary:")
    for name, info in results.items():
        print(f"  {info['name']}: {info['train_time_seconds']:.2f}s")


def cmd_evaluate():
    """多维度评估"""
    print("=" * 60)
    print("STEP 4: MULTI-DIMENSIONAL EVALUATION")
    print("=" * 60)

    train_df, test_df, val_df = load_data()
    processed = preprocess_pipeline(train_df, test_df, val_df)

    algo_names = ["knn", "logistic_regression", "svm", "ann", "xgboost"]
    run_full_evaluation(
        processed["X_train"], processed["y_train"],
        processed["X_val"], processed["y_val"],
        processed["X_test"], processed["y_test"],
        algo_names=algo_names,
    )


def cmd_full_pipeline():
    """一键运行全流程"""
    print("=" * 60)
    print("FULL PIPELINE: Dry Bean ML Project")
    print("=" * 60)

    # 1. 数据分析
    cmd_analyze()

    # 2. 数据预处理
    processed = cmd_preprocess()

    # 3. 训练
    print("\n" + "=" * 60)
    print("STEP 3: MODEL TRAINING")
    print("=" * 60)
    algo_names = ["knn", "logistic_regression", "svm", "ann", "xgboost"]
    results = train_all_algorithms(
        processed["X_train"], processed["y_train"],
        processed["X_val"], processed["y_val"],
        algos=algo_names,
    )
    print("\nTraining Summary:")
    for name, info in results.items():
        print(f"  {info['name']}: {info['train_time_seconds']:.2f}s")

    # 4. 评估
    print("\n" + "=" * 60)
    print("STEP 4: MULTI-DIMENSIONAL EVALUATION")
    print("=" * 60)
    run_full_evaluation(
        processed["X_train"], processed["y_train"],
        processed["X_val"], processed["y_val"],
        processed["X_test"], processed["y_test"],
        algo_names=algo_names,
    )

    print("\n" + "=" * 60)
    print("FULL PIPELINE COMPLETE!")
    print("All outputs saved to outputs/ directory")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Dry Bean Dataset ML Project - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py analyze        # Data analysis + charts
  python main.py preprocess     # Data preprocessing
  python main.py train          # Train all algorithms
  python main.py evaluate       # Multi-dimension evaluation
  python main.py full-pipeline  # Run everything
        """,
    )
    parser.add_argument("command", nargs="?",
                        choices=["analyze", "preprocess", "train", "evaluate", "full-pipeline"],
                        default="full-pipeline",
                        help="Command to execute")

    args = parser.parse_args()

    if args.command == "analyze":
        cmd_analyze()
    elif args.command == "preprocess":
        cmd_preprocess()
    elif args.command == "train":
        cmd_train()
    elif args.command == "evaluate":
        cmd_evaluate()
    elif args.command == "full-pipeline":
        cmd_full_pipeline()


if __name__ == "__main__":
    main()
