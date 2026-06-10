"""评估模块 - 精度、Loss曲线、推理速度、鲁棒性、过拟合分析"""
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

from .algorithms.base import AlgorithmFactory
from .algorithms import logistic_regression, svm_classifier, knn_classifier
from .algorithms import xgboost_classifier, ann_classifier
from .trainer import load_trained_model, train_single_algorithm
from .noise_generator import generate_robustness_test_data
from . import visualizer

RESULTS_DIR = "outputs/results"
FIGURES_DIR = "outputs/figures"


def evaluate_accuracy(clf, X_test, y_test):
    """计算精度指标"""
    y_pred = clf.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }


def evaluate_inference_speed(clf, X_test, n_runs=100):
    """评估推理速度"""
    # 单样本推理
    x_single = X_test.iloc[:1]
    times_single = []
    for _ in range(min(n_runs, 50)):
        t0 = time.time()
        clf.predict(x_single)
        times_single.append((time.time() - t0) * 1000)
    avg_single_ms = np.mean(times_single)

    # 全量推理
    t0 = time.time()
    clf.predict(X_test)
    total_time_ms = (time.time() - t0) * 1000

    return {
        "single_inference_ms": avg_single_ms,
        "total_inference_ms": total_time_ms,
        "batch_inference_ms_per_sample": total_time_ms / len(X_test),
    }


def evaluate_overfitting(clf, X_train, y_train, X_test, y_test):
    """过拟合分析"""
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "gap": train_acc - test_acc,
    }


def evaluate_robustness(X_test, y_test, classes, algo_names):
    """鲁棒性分析：不同噪声下的精度下降"""
    noise_configs = [
        {"type": "gaussian", "level": 0.01},
        {"type": "gaussian", "level": 0.05},
        {"type": "gaussian", "level": 0.1},
        {"type": "label", "level": 0.05},
        {"type": "label", "level": 0.10},
        {"type": "label", "level": 0.20},
        {"type": "missing", "level": 0.05},
        {"type": "missing", "level": 0.10},
    ]

    all_noisy_datasets = generate_robustness_test_data(X_test, y_test, noise_configs)

    # 按噪声类型组织结果
    robustness = {
        "Gaussian Noise": {},
        "Label Noise": {},
        "Missing Values": {},
    }

    for algo_name in algo_names:
        clf = load_trained_model(algo_name)

        for cfg in noise_configs:
            label = f"{cfg['type']}_{cfg['level']}"
            X_n, y_n = all_noisy_datasets[label]
            try:
                acc = accuracy_score(y_n, clf.predict(X_n))
            except Exception:
                acc = 0.0

            noise_type_map = {"gaussian": "Gaussian Noise", "label": "Label Noise", "missing": "Missing Values"}
            noise_type = noise_type_map[cfg["type"]]
            if algo_name not in robustness[noise_type]:
                robustness[noise_type][algo_name] = {}
            robustness[noise_type][algo_name][cfg["level"]] = acc

    return robustness


def run_full_evaluation(X_train, y_train, X_val, y_val, X_test, y_test,
                        algo_names=None, force_retrain=False):
    """完整评估流程"""
    if algo_names is None:
        algo_names = AlgorithmFactory.list_algorithms()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    all_results = {}
    class_names = sorted(y_train.unique().tolist())

    # 确保所有模型已训练
    for algo_name in algo_names:
        model_path = f"outputs/models/{algo_name}.pkl"
        if force_retrain or not os.path.exists(model_path):
            print(f"Training {algo_name}...")
            train_single_algorithm(algo_name, X_train, y_train, X_val, y_val)

    # --- 1. 精度对比 ---
    print("\n[1/5] Evaluating accuracy...")
    all_cms = {}
    for algo_name in algo_names:
        clf = load_trained_model(algo_name)
        metrics = evaluate_accuracy(clf, X_test, y_test)
        overfit = evaluate_overfitting(clf, X_train, y_train, X_test, y_test)
        speed = evaluate_inference_speed(clf, X_test)
        all_results[algo_name] = {**metrics, **overfit, **speed, "name": clf.name}

        # 混淆矩阵
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=class_names)
        all_cms[algo_name] = cm

    # 精度对比表
    accuracy_table = pd.DataFrame({
        algo: {
            "Accuracy": all_results[algo]["accuracy"],
            "Precision(Macro)": all_results[algo]["precision_macro"],
            "Recall(Macro)": all_results[algo]["recall_macro"],
            "F1(Macro)": all_results[algo]["f1_macro"],
        } for algo in algo_names
    }).T
    accuracy_table.to_csv(os.path.join(RESULTS_DIR, "accuracy_comparison.csv"))
    print(accuracy_table)

    accuracy_dict = {r["name"]: {"test_accuracy": r["accuracy"]} for r in all_results.values()}
    visualizer.plot_accuracy_comparison(
        accuracy_dict, os.path.join(FIGURES_DIR, "evaluation/accuracy_comparison.png")
    )
    visualizer.plot_confusion_matrices(
        all_cms, class_names, os.path.join(FIGURES_DIR, "evaluation/confusion_matrices.png")
    )

    # --- 2. Loss 曲线对比 ---
    print("\n[2/5] Comparing loss curves...")
    loss_records = {}
    for algo_name in algo_names:
        clf = load_trained_model(algo_name)
        loss_records[clf.name] = {}
        train_loss = clf.get_train_losses()
        val_loss = clf.get_val_losses()
        if train_loss is not None and len(train_loss) > 0:
            loss_records[clf.name]["train_loss"] = train_loss
        if val_loss is not None and len(val_loss) > 0:
            loss_records[clf.name]["val_loss"] = val_loss
    visualizer.plot_loss_curves(
        loss_records, os.path.join(FIGURES_DIR, "evaluation/loss_curves.png")
    )

    # --- 3. 推理速度对比 ---
    print("\n[3/5] Comparing inference speed...")
    visualizer.plot_inference_speed(
        {r["name"]: {"inference_time_ms": r["total_inference_ms"]}
         for r in all_results.values()},
        os.path.join(FIGURES_DIR, "evaluation/inference_speed.png")
    )

    # --- 4. 鲁棒性分析 ---
    print("\n[4/5] Evaluating robustness...")
    robustness = evaluate_robustness(X_test, y_test, class_names, algo_names)
    visualizer.plot_robustness(
        robustness, os.path.join(FIGURES_DIR, "evaluation/robustness.png")
    )
    # 保存鲁棒性数据
    robustness_flat = {}
    for noise_type, algos_data in robustness.items():
        for algo, levels in algos_data.items():
            if algo not in robustness_flat:
                robustness_flat[algo] = {}
            robustness_flat[algo][noise_type] = levels
    pd.DataFrame(robustness_flat).to_csv(os.path.join(RESULTS_DIR, "robustness.csv"))

    # --- 5. 过拟合分析 ---
    print("\n[5/5] Analyzing overfitting...")
    train_accs = {all_results[a]["name"]: all_results[a]["train_accuracy"] for a in algo_names}
    test_accs = {all_results[a]["name"]: all_results[a]["test_accuracy"] for a in algo_names}
    visualizer.plot_overfitting(
        train_accs, test_accs,
        os.path.join(FIGURES_DIR, "evaluation/overfitting.png")
    )

    # --- 保存完整结果 ---
    results_json = {}
    for algo_name in algo_names:
        r = all_results[algo_name]
        results_json[algo_name] = {
            "name": r["name"],
            "test_accuracy": float(r["accuracy"]),
            "precision_macro": float(r["precision_macro"]),
            "recall_macro": float(r["recall_macro"]),
            "f1_macro": float(r["f1_macro"]),
            "train_accuracy": float(r["train_accuracy"]),
            "overfitting_gap": float(r["gap"]),
            "train_time_seconds": float(r.get("train_time_seconds", 0)),
            "inference_total_ms": float(r["total_inference_ms"]),
            "inference_per_sample_ms": float(r["batch_inference_ms_per_sample"]),
        }
    with open(os.path.join(RESULTS_DIR, "full_results.json"), "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    # 打印汇总表
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    summary_cols = ["test_accuracy", "overfitting_gap", "inference_total_ms"]
    for algo_name, r in results_json.items():
        print(f"\n{r['name']}:")
        print(f"  Test Accuracy:     {r['test_accuracy']:.4f}")
        print(f"  Train Accuracy:    {r['train_accuracy']:.4f}")
        print(f"  Overfitting Gap:   {r['overfitting_gap']:.4f}")
        print(f"  F1 (Macro):        {r['f1_macro']:.4f}")
        print(f"  Inference Time:    {r['inference_total_ms']:.2f} ms")

    return all_results
