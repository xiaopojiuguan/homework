# Dry Bean Dataset - 机器学习全流程工程项目

基于 UCI Dry Bean Dataset 的多分类机器学习全流程工程，涵盖数据分析、数据清洗、多算法实验对比、系统集成。

## 项目概述

本项目对 Dry Bean Dataset（含数据污染的脏数据版本）进行完整的机器学习流程处理，实现 7 种干豆的分类任务。

### 数据集描述

- **来源**: UCI Machine Learning Repository - Dry Bean Dataset
- **样本量**: Train 9,527 / Test 2,737 / Val 1,347
- **特征**: 16 个数值型特征（面积、周长、长轴长度、短轴长度、纵横比、离心率、凸面积、等效直径、延展度、坚实度、圆度、紧凑度、形状因子1-4）
- **类别**: 7 种豆类
  - DERMASON (Dermason 豆)
  - SIRA (Sira 豆)
  - SEKER (Seker 豆)
  - HOROZ (Horoz 豆)
  - CALI (Cali 豆)
  - BARBUNYA (Barbunya 豆)
  - BOMBAY (Bombay 豆)

### 数据污染情况

- **缺失值**: Perimeter 和 Solidity 字段存在缺失值（训练集约 5%）
- **标签噪声**: 原始标签存在拼写错误（D3RMAS0N、S3K3R 等）、大小写混乱、尾部空格等

## 数据处理方法

### 1. 标签清洗
- 去除首尾空格
- 统一转为大写
- 错误拼写纠正映射（如 D3RMAS0N → DERMASON, H0R0Z → HOROZ）

### 2. 缺失值处理
- 按类别分组，使用中位数填充 Perimeter 和 Solidity
- 对于类别未知的样本，回退到全局中位数

### 3. 特征工程
- StandardScaler 标准化（均值为0，方差为1）
- 保证训练集、测试集、验证集使用相同的缩放参数

## 实现的算法

| 算法 | 类型 | 来源 | 说明 |
|------|------|------|------|
| KNN | 距离度量 | scikit-learn | 课内算法，k=5 近邻投票 |
| Logistic Regression | 线性模型 | scikit-learn | 课内算法，multinomial 多分类 |
| SVM | 支持向量机 | scikit-learn | 课内算法，RBF 核，支持概率输出 |
| **Hand-written ANN** | 神经网络 | **从零手写** | 课内算法，手写前向传播 + 反向传播，ReLU + Softmax |
| **XGBoost** | 梯度提升 | xgboost | **课外算法**，支持 early stopping |

> ANN 手写实现完全从零构建：He 初始化 → Forward Propagation → Cross-Entropy Loss → Backpropagation → Gradient Descent，未使用 PyTorch/TensorFlow。

## 实验结果

### 测试集精度对比

| 算法 | Accuracy | Precision(Macro) | Recall(Macro) | F1(Macro) | 训练集Acc | 过拟合Gap | 推理时间(ms) |
|------|----------|-----------------|---------------|-----------|-----------|-----------|-------------|
| KNN | 0.9204 | 0.9328 | 0.9261 | 0.9293 | 0.9415 | 0.0212 | 63.22 |
| Logistic Regression | 0.9214 | 0.9339 | 0.9292 | 0.9312 | 0.9272 | 0.0057 | ~0.00 |
| SVM | 0.9309 | 0.9416 | 0.9364 | 0.9390 | 0.9309 | -0.0000 | 239.05 |
| Hand-written ANN | 0.9291 | 0.9388 | 0.9362 | 0.9374 | 0.9367 | 0.0076 | ~0.00 |
| **XGBoost (课外)** | **0.9317** | **0.9425** | **0.9408** | **0.9416** | 0.9758 | 0.0441 | 5.76 |

> 上表由 `python main.py full-pipeline` 自动生成，详见 `outputs/results/accuracy_comparison.csv`

> 上表结果由 `python main.py evaluate` 自动生成，详见 `outputs/results/accuracy_comparison.csv`

### 分析维度

1. **测试集精度对比** — Accuracy, Precision, Recall, F1-Score + 混淆矩阵
2. **Loss 曲线对比** — 训练过程中的损失函数下降趋势
3. **推理速度对比** — 单样本和批量推理时间
4. **鲁棒性分析** — 高斯噪声、标签噪声、缺失值下的精度退化
5. **过拟合分析** — 训练集与测试集精度差异

## 工程架构

```
project/
├── data/                              # 数据集
│   ├── Dry_Bean_Dataset_Dirty_train.csv
│   ├── Dry_Bean_Dataset_Dirty_test.csv
│   └── Dry_Bean_Dataset_Dirty_val.csv
├── src/                               # 源代码
│   ├── data_loader.py                 # 数据加载模块
│   ├── preprocessor.py                # 预处理模块
│   ├── visualizer.py                  # 可视化模块
│   ├── noise_generator.py             # 噪声注入模块
│   ├── trainer.py                     # 训练调度模块
│   ├── evaluator.py                   # 评估分析模块
│   └── algorithms/                    # 算法模块
│       ├── base.py                    # 基类 + 工厂模式
│       ├── logistic_regression.py     # 逻辑回归
│       ├── random_forest.py           # 随机森林
│       ├── xgboost_classifier.py      # XGBoost
│       └── svm_classifier.py          # SVM
├── outputs/                           # 输出目录
│   ├── figures/                       # 图表
│   ├── models/                        # 模型文件
│   └── results/                       # 实验结果
├── main.py                            # CLI 统一入口
├── requirements.txt                   # 依赖
└── README.md                          # 项目说明
```

## 快速开始

### 环境要求

- Python >= 3.8
- pip

### 安装

```bash
pip install -r requirements.txt
```

### 运行

```bash
# 数据分析
python main.py analyze

# 数据预处理
python main.py preprocess

# 训练所有算法
python main.py train

# 多维度评估
python main.py evaluate

# 一键运行全流程
python main.py full-pipeline
```

### 输出

所有结果保存在 `outputs/` 目录下：
- `figures/data_analysis/` — 数据分析图表
- `figures/evaluation/` — 评估对比图表
- `models/` — 训练好的模型（.pkl）
- `results/` — 数值结果（CSV + JSON）

## 课程总结

通过本课程的学习，掌握了机器学习项目从数据到部署的完整流程：
1. 数据分析与可视化方法
2. 数据清洗与特征工程技巧
3. 多种分类算法的原理与实现
4. 模型评估的多维度分析方法
5. 工程化项目的组织与交付

## License

MIT
