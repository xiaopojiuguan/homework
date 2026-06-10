"""可视化模块 - 生成所有分析图表"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "outputs/figures"


def set_style():
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 150


def plot_class_distribution(y, title, save_path):
    """类别分布柱状图"""
    set_style()
    counts = y.value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(counts.index, counts.values, color=sns.color_palette("Set2", len(counts)))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", fontsize=9)
    plt.xticks(rotation=30)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_distributions(df, class_col, save_dir):
    """特征分布直方图"""
    set_style()
    os.makedirs(save_dir, exist_ok=True)
    features = [c for c in df.columns if c != class_col]
    n_features = len(features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    axes = axes.flatten()
    for i, feat in enumerate(features):
        for cls in df[class_col].unique():
            subset = df[df[class_col] == cls][feat]
            subset = pd.to_numeric(subset, errors="coerce").dropna().values
            if len(subset) > 0:
                axes[i].hist(subset, bins=30, alpha=0.5, label=cls)
        axes[i].set_title(feat, fontsize=9)
        axes[i].tick_params(labelsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=7)
    plt.suptitle("Feature Distributions by Class", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(os.path.join(save_dir, "feature_distributions.png"), bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df, save_path):
    """相关性热力图"""
    set_style()
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_null_heatmap(df, title, save_path):
    """缺失值热力图"""
    set_style()
    null_mask = df.isnull()
    fig, ax = plt.subplots(figsize=(14, 2))
    sns.heatmap(null_mask.T, cmap=["#2ecc71", "#e74c3c"], cbar=False,
                xticklabels=False, ax=ax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_comparison(results_dict, save_path):
    """算法精度对比柱状图"""
    set_style()
    names = list(results_dict.keys())
    accuracies = [results_dict[n]["test_accuracy"] for n in names]
    colors = sns.color_palette("Set2", len(names))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, accuracies, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Test Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.4f}", ha="center", fontsize=11, fontweight="bold")
    plt.xticks(rotation=15)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(loss_records, save_path):
    """Loss 曲线对比"""
    set_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, records in loss_records.items():
        if records is not None and "train_loss" in records:
            epochs = range(1, len(records["train_loss"]) + 1)
            ax.plot(epochs, records["train_loss"], "-", label=f"{name} (train)", linewidth=1.5)
            if "val_loss" in records:
                ax.plot(epochs, records["val_loss"], "--", label=f"{name} (val)", linewidth=1.5)
    ax.set_title("Loss Curves Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_inference_speed(results_dict, save_path):
    """推理速度对比"""
    set_style()
    names = list(results_dict.keys())
    times = [results_dict[n].get("inference_time_ms", 0) for n in names]
    colors = sns.color_palette("Set2", len(names))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, times, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Inference Speed Comparison (ms)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Time (ms)")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times) * 0.01,
                f"{t:.2f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_robustness(robustness_results, save_path):
    """鲁棒性对比：噪声强度 vs 精度下降"""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (noise_type, data) in enumerate(robustness_results.items()):
        ax = axes[idx]
        x_labels = []
        for algo_name, points in data.items():
            levels = sorted(points.keys())
            x_labels = [str(l) for l in levels]
            accs = [points[l] for l in levels]
            ax.plot(x_labels, accs, "-o", label=algo_name, markersize=6, linewidth=1.5)
        ax.set_title(noise_type, fontsize=12, fontweight="bold")
        ax.set_xlabel("Noise Level")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Robustness Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_overfitting(train_accs, test_accs, save_path):
    """过拟合分析：训练精度 vs 测试精度"""
    set_style()
    names = list(train_accs.keys())
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, [train_accs[n] for n in names], width,
                   label="Train Accuracy", color="#3498db", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, [test_accs[n] for n in names], width,
                   label="Test Accuracy", color="#e74c3c", edgecolor="black", linewidth=0.5)
    ax.set_title("Overfitting Analysis: Train vs Test Accuracy", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.legend()
    # 标注差异
    for i, name in enumerate(names):
        gap = train_accs[name] - test_accs[name]
        y_pos = max(train_accs[name], test_accs[name]) + 0.005
        ax.annotate(f"gap={gap:.3f}", (x[i], y_pos), ha="center", fontsize=9,
                    color="red" if gap > 0.05 else "green")
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(all_cms, class_names, save_path):
    """混淆矩阵汇总"""
    set_style()
    n = len(all_cms)
    n_cols = min(2, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    if n == 1:
        axes = [axes]
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (name, cm) in enumerate(all_cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"size": 7})
        axes[i].set_title(f"{name} Confusion Matrix", fontsize=12, fontweight="bold")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
