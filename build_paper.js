const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, LevelFormat, ExternalHyperlink,
  ImageRun, TabStopType, TabStopPosition
} = require("docx");

const PROJECT = "D:/Desktop/3/期末大作业";
const OUT_FIG = `${PROJECT}/outputs/figures`;

// Helper functions
const border = { style: BorderStyle.SINGLE, size: 1, color: "AAAAAA" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };
const headerShading = { fill: "1F4E79", type: ShadingType.CLEAR };
const altShading = { fill: "F2F7FB", type: ShadingType.CLEAR };

function headerCell(text, width) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA }, margins: cellMargins,
    shading: headerShading,
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text, bold: true, font: "Microsoft YaHei", size: 20, color: "FFFFFF" })] })]
  });
}

function dataCell(text, width, shade = false) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA }, margins: cellMargins,
    shading: shade ? altShading : undefined,
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: String(text), font: "Microsoft YaHei", size: 18 })] })]
  });
}

function heading1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text, bold: true, font: "Microsoft YaHei", size: 32 })] });
}

function heading2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text, bold: true, font: "Microsoft YaHei", size: 28 })] });
}

function heading3(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text, bold: true, font: "Microsoft YaHei", size: 24 })] });
}

function bodyText(text) {
  return new Paragraph({ spacing: { before: 60, after: 60 }, children: [new TextRun({ text, font: "Microsoft YaHei", size: 22 })] });
}

function bodyTextIndent(text) {
  return new Paragraph({ spacing: { before: 40, after: 40 }, indent: { firstLine: 480 }, children: [new TextRun({ text, font: "Microsoft YaHei", size: 22 })] });
}

function figure(path, width, height, caption) {
  const imgs = [];
  if (fs.existsSync(path)) {
    imgs.push(new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width, height }, altText: { title: caption, description: caption, name: "fig" } })]
    }));
  }
  imgs.push(new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new TextRun({ text: caption, font: "Microsoft YaHei", size: 18, italics: true, color: "555555" })]
  }));
  return imgs;
}

// ===== BUILD DOCUMENT =====
const children = [];

// ===== COVER PAGE =====
children.push(
  new Paragraph({ spacing: { before: 3000 } }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "机器学习期末大作业", bold: true, font: "Microsoft YaHei", size: 48, color: "1F4E79" })] }),
  new Paragraph({ spacing: { after: 200 } }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "基于 Dry Bean Dataset 的多分类\n全流程机器学习工程", font: "Microsoft YaHei", size: 30, color: "333333" })] }),
  new Paragraph({ spacing: { after: 600 } }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "——数据分析 · 数据处理 · 多算法实验 · 系统展示 · 课程总结", font: "Microsoft YaHei", size: 22, color: "666666" })] }),
  new Paragraph({ spacing: { before: 1600 } }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "课程：机器学习", font: "Microsoft YaHei", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "提交日期：2026年6月", font: "Microsoft YaHei", size: 24 })] }),
  new Paragraph({ pageBreakAfter: true })
);

// ===== TABLE OF CONTENTS (simplified) =====
children.push(
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "目  录", bold: true, font: "Microsoft YaHei", size: 32 })] }),
  bodyText("第一章  数据分析"),
  bodyText("第二章  数据处理"),
  bodyText("第三章  多算法实验分析"),
  bodyText("    3.1  算法介绍"),
  bodyText("    3.2  测试集精度对比"),
  bodyText("    3.3  Loss曲线对比"),
  bodyText("    3.4  推理速度对比"),
  bodyText("    3.5  鲁棒性分析"),
  bodyText("    3.6  过拟合分析"),
  bodyText("第四章  系统展示"),
  bodyText("第五章  课程总结"),
  bodyText("参考文献"),
  new Paragraph({ pageBreakAfter: true })
);

// ===== CHAPTER 1: 数据分析 =====
children.push(
  new Paragraph({ pageBreakBefore: true }),
  heading1("第一章  数据分析"),

  heading2("1.1  数据集概述"),
  bodyTextIndent("本实验使用的数据集为 Dry Bean Dataset（干豆数据集），来源于 UCI Machine Learning Repository。该数据集包含 7 种不同类型的干豆的形态学特征，通过计算机视觉系统从干豆图像中提取。数据集的原始特征包括面积（Area）、周长（Perimeter）、长轴长度（MajorAxisLength）、短轴长度（MinorAxisLength）、纵横比（AspectRation）、离心率（Eccentricity）、凸面积（ConvexArea）、等效直径（EquivDiameter）、延展度（Extent）、坚实度（Solidity）、圆度（roundness）、紧凑度（Compactness）和四个形状因子（ShapeFactor1-4），共计 16 个数值型特征。"),
  bodyTextIndent("教师已将数据集预先划分为训练集（Train）、测试集（Test）和验证集（Val）三个子集，具体样本量如下表所示："),

  // Data split table
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [3120, 2080, 2080, 2080],
    rows: [
      new TableRow({ children: [headerCell("数据集", 3120), headerCell("样本量", 2080), headerCell("特征数", 2080), headerCell("类别数", 2080)] }),
      new TableRow({ children: [dataCell("训练集 (Train)", 3120, true), dataCell("9,527", 2080, true), dataCell("16", 2080, true), dataCell("25（原始）", 2080, true)] }),
      new TableRow({ children: [dataCell("测试集 (Test)", 3120), dataCell("2,737", 2080), dataCell("16", 2080), dataCell("24（原始）", 2080)] }),
      new TableRow({ children: [dataCell("验证集 (Val)", 3120, true), dataCell("1,347", 2080, true), dataCell("16", 2080, true), dataCell("24（原始）", 2080, true)] }),
    ]
  }),
  new Paragraph({ spacing: { after: 100 } }),

  bodyTextIndent("数据集的目标类别为 7 种干豆：DERMASON（Dermason豆）、SIRA（Sira豆）、SEKER（Seker豆）、HOROZ（Horoz豆）、CALI（Cali豆）、BARBUNYA（Barbunya豆）和 BOMBAY（Bombay豆）。类别分布存在一定程度的不均衡，DERMASON 类最多，BOMBAY 类最少。"),

  ...figure(`${OUT_FIG}/data_analysis/class_distribution.png`, 440, 220, "图1-1 训练集原始类别分布柱状图"),
  ...figure(`${OUT_FIG}/data_analysis/feature_distributions.png`, 500, 400, "图1-2 各特征在各类别下的分布直方图"),
  ...figure(`${OUT_FIG}/data_analysis/correlation_heatmap.png`, 440, 380, "图1-3 特征相关性热力图"),

  heading2("1.2  数据污染分析"),
  bodyTextIndent("教师提供的数据为\"脏数据\"版本（Dirty Version），包含多种类型的数据污染，需要仔细分析后才能进行后续的清洗工作。经过详细的数据探查，发现了以下三类主要的污染问题："),

  heading3("1.2.1  缺失值"),
  bodyTextIndent("数据集中存在两种类型的缺失值：标准缺失值（NaN）和以特殊符号\"?\"标记的缺失值。具体而言，Perimeter（周长）和 Solidity（坚实度）两个字段存在缺失。在训练集中，Perimeter 有 469 个缺失值（占比约 4.9%），Solidity 有 474 个缺失值（含 202 个\"?\"标记，占比约 5.0%）。测试集和验证集中也存在类似比例的缺失值。"),

  ...figure(`${OUT_FIG}/data_analysis/null_heatmap.png`, 500, 80, "图1-4 训练集缺失值热力图（红色为缺失）"),

  heading3("1.2.2  标签噪声"),
  bodyTextIndent("原始数据集中类别标签（Class字段）存在严重的质量问题。清洗前共有25种不同的标签值，而实际应有7种。标签噪声的具体表现形式包括："),
  bodyTextIndent("（1）拼写错误：如 D3RMAS0N（应为 DERMASON）、S3K3R（应为 SEKER）、H0R0Z（应为 HOROZ）、B0MBAY（应为 BOMBAY），其中数字 0 替换了字母 O，数字 3 替换了字母 E。这些拼写变体共涉及约 116 个训练样本。"),
  bodyTextIndent("（2）大小写混乱：存在以小写字母书写的标签，如 dermason、sira、horoz、cali、seker、barbunya、bombay，共涉及约 213 个训练样本。"),
  bodyTextIndent("（3）尾部空格：部分标签末尾包含了多余的空格字符，如\"DERMASON \"、\"SIRA \"、\"HOROZ \"等，共涉及约 93 个训练样本。"),

  heading3("1.2.3  数值数据中的非数值内容"),
  bodyTextIndent("Compactness（紧凑度）字段中存在 258 个包含单位后缀\"cm\"的值（如\"0.9293 cm\"），导致该列被 pandas 识别为字符串类型而非数值类型。根据数据集文档，Compactness 应为无量纲比值，\"cm\"标记属于数据采集过程中的标注错误，需要去除单位并转换为数值。"),
);

// ===== CHAPTER 2: 数据处理 =====
children.push(
  new Paragraph({ pageBreakBefore: true }),
  heading1("第二章  数据处理"),

  heading2("2.1  标签清洗"),
  bodyTextIndent("标签清洗采用三步流水线处理，确保所有标签统一到标准的7类别系统："),
  bodyTextIndent("第一步：去除首尾空格。使用 Python 的 str.strip() 方法，消除尾部空格和首部空格对标签匹配的干扰。"),
  bodyTextIndent("第二步：统一转换为大写。使用 str.upper() 将所有标签统一为大写格式，消除大小写混乱问题。"),
  bodyTextIndent("第三步：拼写错误纠正。建立错误标签到正确标签的映射字典（D3RMAS0N→DERMASON, S3K3R→SEKER, H0R0Z→HOROZ, B0MBAY→BOMBAY），使用该映射对标签进行批量替换。"),
  bodyTextIndent("清洗完成后，标签类别从原始的 25 种降至标准的 7 种，各类别在训练集中的分布为：DERMASON(2503)、SIRA(1837)、SEKER(1408)、HOROZ(1340)、CALI(1151)、BARBUNYA(927)、BOMBAY(361)。"),

  heading2("2.2  缺失值处理"),
  bodyTextIndent("缺失值处理分为两个步骤。首先，在数据加载阶段将\"?\"标记统一读取为标准的 NaN 值（通过 pandas.read_csv 的 na_values 参数）。然后，对 Perimeter 和 Solidity 两列的缺失值采用按类别分组的中位数填充策略："),
  bodyTextIndent("相比于全局中位数填充，按类别分组填充保留了类内数据的统计特性，避免了不同类别之间的数据混淆。具体实现为：对每个样本，使用其所属类别的中位数填充缺失值；若该类别未能提供有效中位数（所有样本均缺失），则回退到全局中位数。"),
  bodyTextIndent("对于测试集和验证集，采用相同的策略进行独立填充，避免使用训练集的信息导致数据泄露。"),

  heading2("2.3  数值清理"),
  bodyTextIndent("针对 Compactness 列中的单位后缀问题，设计正则表达式去除常见的单位标记（cm、mm、px、in等）：str.replace(r\"\\s*(cm|mm|px|in)$\", \"\", regex=True)，然后使用 pd.to_numeric() 将所有列强制转换为数值类型，无法转换的值自动变为 NaN 并由缺失值填充步骤统一处理。"),

  heading2("2.4  特征标准化"),
  bodyTextIndent("由于不同特征的量纲差异较大（如 Area 范围 10000~300000，而 roundness 范围 0~1），直接使用这些特征进行模型训练可能导致梯度下降过程不稳定，或者基于距离的算法（如 KNN、SVM）被大数值特征主导。"),
  bodyTextIndent("因此，使用 sklearn 的 StandardScaler 对所有特征进行 Z-score 标准化（均值为 0，方差为 1）。标准化器仅在训练集上拟合（fit），然后对测试集和验证集使用相同的参数进行变换（transform），确保三个数据集的特征空间一致。"),

  new Paragraph({ spacing: { after: 100 } }),

  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [3120, 3120, 3120],
    rows: [
      new TableRow({ children: [headerCell("处理步骤", 3120), headerCell("方法", 3120), headerCell("效果", 3120)] }),
      new TableRow({ children: [dataCell("标签清洗", 3120, true), dataCell("去空格+大写+纠错映射", 3120, true), dataCell("25种→7种标准标签", 3120, true)] }),
      new TableRow({ children: [dataCell("缺失值填充", 3120), dataCell("按类别分组中位数", 3120), dataCell("消除所有缺失值", 3120)] }),
      new TableRow({ children: [dataCell("数值清理", 3120, true), dataCell("正则去单位+数值转换", 3120, true), dataCell("Compactness恢复为数值", 3120, true)] }),
      new TableRow({ children: [dataCell("特征标准化", 3120, true), dataCell("StandardScaler", 3120, true), dataCell("均值0、方差1", 3120, true)] }),
    ]
  }),
  new Paragraph({ spacing: { after: 60 } }),
  bodyText("表2-1 数据处理流水线总结"),
);

// ===== CHAPTER 3: 多算法实验分析 =====
children.push(
  new Paragraph({ pageBreakBefore: true }),
  heading1("第三章  多算法实验分析"),

  heading2("3.1  算法介绍"),
  bodyTextIndent("本实验共实现并对比了 5 种多分类算法，其中 4 种为课堂上学习过的算法（KNN、逻辑回归、SVM、ANN），1 种为课外自学算法（XGBoost）。特别地，ANN（人工神经网络）按照课程要求从零手写实现，包含前向传播和反向传播算法，未使用任何深度学习框架。"),

  heading3("3.1.1  KNN（K近邻分类器）"),
  bodyTextIndent("KNN 是一种基于距离度量的非参数分类算法，课内算法。通过计算测试样本与所有训练样本之间的距离，选取最近的 K 个邻居进行多数投票。本实验设置 K=5，采用欧氏距离。KNN 无需训练过程，但在推理时需要计算与所有训练样本的距离，因此随着数据量增大推理时间显著增加。"),

  heading3("3.1.2  Logistic Regression（逻辑回归）"),
  bodyTextIndent("逻辑回归是一种经典的线性分类模型，课内算法。通过 softmax 函数将线性预测值映射为多类概率分布。本实验使用 sklearn 的 LogisticRegression，采用 multinomial 多分类模式、lbfgs 求解器，最大迭代次数设为 5000。由于使用 L2 正则化，模型具有较好的泛化能力。"),

  heading3("3.1.3  SVM（支持向量机）"),
  bodyTextIndent("SVM 通过寻找最大化分类间隔的超平面来进行分类，课内算法。本实验使用 sklearn 的 SVC，采用 RBF（径向基函数）核，支持概率输出（probability=True）。RBF 核使得 SVM 能够处理非线性可分问题，但由于需要对每对样本计算核函数，推理速度较慢。"),

  heading3("3.1.4  Hand-written ANN（手写人工神经网络）"),
  bodyTextIndent("ANN 是本课程重点学习的内容，按照课堂要求\"手写ANN（前向+BP）\"，从零实现了一个多层感知机（MLP），完全基于 NumPy 编写，未使用 PyTorch、TensorFlow 等任何深度学习框架。"),
  bodyTextIndent("网络结构：输入层（16个神经元，对应16个特征）→ 隐藏层1（128个神经元，ReLU激活）→ 隐藏层2（64个神经元，ReLU激活）→ 输出层（7个神经元，Softmax激活）。"),
  bodyTextIndent("核心实现包括以下组件："),
  bodyTextIndent("（1）权重初始化：采用 He 初始化策略 W ~ N(0, sqrt(2/n_in))，有效缓解深层网络的梯度消失问题。"),
  bodyTextIndent("（2）前向传播（Forward Propagation）：逐层计算 Z = W·A_prev + b，然后通过 ReLU 激活函数 A = max(0, Z)，最后通过 Softmax 函数将输出归一化为概率分布：softmax(z_i) = exp(z_i) / Σexp(z_j)。"),
  bodyTextIndent("（3）损失函数：多分类交叉熵损失 L = -(1/m)·ΣΣ y_ij · log(p_ij)。"),
  bodyTextIndent("（4）反向传播（Backpropagation）：从输出层开始逐层计算梯度——输出层梯度 ∂L/∂Z = (y_pred - y_true)/m，隐藏层梯度通过链式法则传播，ReLU 的导数为 ∂ReLU/∂Z = 1(Z > 0)。"),
  bodyTextIndent("（5）参数更新：采用 Mini-batch 梯度下降，batch_size=64，learning_rate=0.01，epochs=100。"),

  heading3("3.1.5  XGBoost（极端梯度提升）⭐课外算法"),
  bodyTextIndent("XGBoost（eXtreme Gradient Boosting）是一种基于梯度提升框架的集成学习算法，为本次实验的课外自学算法，课堂上未讲授。XGBoost 通过逐步添加决策树来优化目标函数，每棵新树拟合前一步的残差。相比于传统的梯度提升方法，XGBoost 引入了以下关键改进："),
  bodyTextIndent("（1）正则化目标函数：在损失函数中同时加入 L1 和 L2 正则项，有效控制模型复杂度，抑制过拟合。"),
  bodyTextIndent("（2）二阶泰勒展开：利用损失函数的二阶导数信息，比传统 GBDT 仅使用一阶导数能更精确地逼近最优解。"),
  bodyTextIndent("（3）列采样和缩减（Shrinkage）：类似随机森林的列采样策略增加基学习器的多样性，通过学习率缩减每棵树的贡献防止过拟合。"),
  bodyTextIndent("（4）并行化计算：在特征粒度上对分裂点查找进行并行化，显著提升训练效率。"),
  bodyTextIndent("本实验使用 xgboost 库的 XGBClassifier，设置 n_estimators=200、max_depth=6、learning_rate=0.1，并启用 early_stopping（20轮无改善即停止），防止过拟合。"),

  heading2("3.2  测试集精度对比"),
  bodyTextIndent("在统一的数据预处理流水线后，各算法在测试集上的精度表现如下表所示："),

  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2340, 1404, 1404, 1404, 1404, 1404],
    rows: [
      new TableRow({ children: [
        headerCell("算法", 2340), headerCell("Accuracy", 1404), headerCell("Precision(M)", 1404),
        headerCell("Recall(M)", 1404), headerCell("F1(Macro)", 1404), headerCell("训练时间", 1404)
      ] }),
      new TableRow({ children: [
        dataCell("KNN", 2340, true), dataCell("0.9204", 1404, true), dataCell("0.9328", 1404, true),
        dataCell("0.9261", 1404, true), dataCell("0.9293", 1404, true), dataCell("0.02s", 1404, true)
      ] }),
      new TableRow({ children: [
        dataCell("Logistic Regression", 2340), dataCell("0.9214", 1404), dataCell("0.9339", 1404),
        dataCell("0.9292", 1404), dataCell("0.9312", 1404), dataCell("1.68s", 1404)
      ] }),
      new TableRow({ children: [
        dataCell("SVM", 2340, true), dataCell("0.9309", 1404, true), dataCell("0.9416", 1404, true),
        dataCell("0.9364", 1404, true), dataCell("0.9390", 1404, true), dataCell("1.45s", 1404, true)
      ] }),
      new TableRow({ children: [
        dataCell("Hand-written ANN", 2340), dataCell("0.9295", 1404), dataCell("0.9386", 1404),
        dataCell("0.9363", 1404), dataCell("0.9373", 1404), dataCell("3.91s", 1404)
      ] }),
      new TableRow({ children: [
        dataCell("XGBoost (课外)", 2340, true), dataCell("0.9317", 1404, true), dataCell("0.9424", 1404, true),
        dataCell("0.9408", 1404, true), dataCell("0.9416", 1404, true), dataCell("0.75s", 1404, true)
      ] }),
    ]
  }),
  new Paragraph({ spacing: { after: 60 } }),
  bodyText("表3-1 五种算法的测试集精度对比（Macro-averaged metrics）"),

  bodyTextIndent("从测试集精度来看，XGBoost 以 93.17% 的准确率和 0.9416 的 F1-Macro 得分取得最优表现，紧随其后的是 SVM（92.09%）和手写 ANN（92.95%）。值得注意的是，手写 ANN 的精度与 sklearn 内置的 SVM 非常接近（仅相差 0.14%），说明手写实现的正确性和有效性。KNN 和逻辑回归分别以 92.04% 和 92.14% 的准确率位列末尾，但仍然是非常有竞争力的结果。"),

  ...figure(`${OUT_FIG}/evaluation/accuracy_comparison.png`, 420, 210, "图3-1 测试集精度对比柱状图"),
  ...figure(`${OUT_FIG}/evaluation/confusion_matrices.png`, 440, 400, "图3-2 各算法混淆矩阵对比"),

  heading2("3.3  Loss曲线对比"),
  bodyTextIndent("Logistic Regression 和 XGBoost 为迭代训练型算法，记录了训练过程中的损失函数（Loss）变化。Logistic Regression 使用对数损失（Log Loss），XGBoost 使用多分类对数损失（mlogloss）。KNN 为非训练型算法，SVM 和随机森林属于非迭代训练型算法（不产生逐步的 Loss 记录），因此在 Loss 曲线对比中仅展示 LR 和 XGBoost。此外，手写 ANN 记录了每轮训练的交叉熵损失。"),

  ...figure(`${OUT_FIG}/evaluation/loss_curves.png`, 450, 200, "图3-3 训练Loss曲线对比"),
  bodyTextIndent("从 Loss 曲线可以看出，XGBoost 的 Loss 下降速度最快，在约 20 轮后即趋于稳定（得益于 early stopping 机制）。Logistic Regression 和手写 ANN 的 Loss 平滑下降，均表现出了良好的收敛性。"),

  heading2("3.4  推理速度对比"),
  bodyTextIndent("推理速度是评估算法实际部署可行性的重要指标。下表记录了各算法对全部 2737 个测试样本进行推理的总耗时："),

  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [3120, 3120, 3120],
    rows: [
      new TableRow({ children: [headerCell("算法", 3120), headerCell("总推理时间(ms)", 3120), headerCell("单样本(ms)", 3120)] }),
      new TableRow({ children: [dataCell("KNN", 3120, true), dataCell("60.80", 3120, true), dataCell("0.0222", 3120, true)] }),
      new TableRow({ children: [dataCell("Logistic Regression", 3120), dataCell("~0.00", 3120), dataCell("~0.0000", 3120)] }),
      new TableRow({ children: [dataCell("SVM", 3120, true), dataCell("234.73", 3120, true), dataCell("0.0858", 3120, true)] }),
      new TableRow({ children: [dataCell("Hand-written ANN", 3120), dataCell("~0.00", 3120), dataCell("~0.0000", 3120)] }),
      new TableRow({ children: [dataCell("XGBoost", 3120, true), dataCell("4.02", 3120, true), dataCell("0.0015", 3120, true)] }),
    ]
  }),
  new Paragraph({ spacing: { after: 60 } }),
  bodyText("表3-2 推理速度对比"),

  ...figure(`${OUT_FIG}/evaluation/inference_speed.png`, 420, 210, "图3-4 推理速度对比柱状图"),
  bodyTextIndent("推理速度差异显著：逻辑回归和手写 ANN 最快（几乎为0ms），因为它们本质上只是矩阵乘法运算。XGBoost 次之（4.02ms），得益于其优化的 C++ 后端。KNN 较慢（60.80ms），因为需要计算与训练集中所有样本的距离。SVM 最慢（234.73ms），RBF 核需要计算每对测试样本与支持向量之间的核函数值。"),

  heading2("3.5  鲁棒性分析"),
  bodyTextIndent("鲁棒性分析通过在测试数据上施加不同类型和强度的噪声，观察各算法精度的退化程度，评估算法在实际应用中面对数据质量下降时的稳定性。本实验测试了三种噪声类型：高斯噪声（模拟传感器误差）、标签噪声（模拟标注错误）和额外缺失值（模拟数据采集不完整）。"),

  ...figure(`${OUT_FIG}/evaluation/robustness.png`, 500, 280, "图3-5 鲁棒性对比（高斯噪声、标签噪声、缺失值）"),
  bodyTextIndent("实验结果表明：（1）面对高斯噪声，所有算法均表现出极强的鲁棒性，即使噪声标准差达到 0.1，精度下降也仅约 1%，这是因为特征已被标准化，高斯噪声的扰动相对较小。（2）标签噪声对所有算法的打击最为严重，20% 的标签噪声下平均精度下降约 19 个百分点至 73-74%，这是最致命的数据污染类型。（3）缺失值对算法的影响居中，10% 的额外缺失率导致精度下降 1-4 个百分点，其中逻辑回归受影响最大。"),
  bodyTextIndent("值得注意的是，XGBoost 在三种噪声类型下均表现出最佳的鲁棒性，这可能得益于其内在的集成学习机制（多棵树的投票平均天然对噪声有抑制作用）。手写 ANN 也表现出不错的鲁棒性，与 SVM 相当。"),

  heading2("3.6  过拟合分析"),
  bodyTextIndent("过拟合是指模型在训练集上表现良好但在测试集上表现欠佳的现象。通过比较训练集精度和测试集精度的差值（Overfitting Gap）来评估各算法的过拟合程度："),

  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2340, 2340, 2340, 2340],
    rows: [
      new TableRow({ children: [headerCell("算法", 2340), headerCell("训练集Acc", 2340), headerCell("测试集Acc", 2340), headerCell("过拟合Gap", 2340)] }),
      new TableRow({ children: [dataCell("KNN", 2340, true), dataCell("0.9415", 2340, true), dataCell("0.9204", 2340, true), dataCell("0.0212", 2340, true)] }),
      new TableRow({ children: [dataCell("Logistic Regression", 2340), dataCell("0.9272", 2340), dataCell("0.9214", 2340), dataCell("0.0057", 2340)] }),
      new TableRow({ children: [dataCell("SVM", 2340, true), dataCell("0.9309", 2340, true), dataCell("0.9309", 2340, true), dataCell("-0.0000", 2340, true)] }),
      new TableRow({ children: [dataCell("Hand-written ANN", 2340), dataCell("0.9363", 2340), dataCell("0.9295", 2340), dataCell("0.0068", 2340)] }),
      new TableRow({ children: [dataCell("XGBoost", 2340, true), dataCell("0.9758", 2340, true), dataCell("0.9317", 2340, true), dataCell("0.0441", 2340, true)] }),
    ]
  }),
  new Paragraph({ spacing: { after: 60 } }),
  bodyText("表3-3 过拟合分析：训练集与测试集精度对比"),

  ...figure(`${OUT_FIG}/evaluation/overfitting.png`, 420, 210, "图3-6 过拟合分析：训练集vs测试集精度"),
  bodyTextIndent("分析显示，SVM 和逻辑回归几乎不存在过拟合（Gap ≤ 0.006），手写 ANN 也有非常小的过拟合差距（0.0068），说明经过标准化后这些模型的泛化能力良好。XGBoost 表现出最明显的过拟合倾向（Gap = 0.0441），即训练集精度（97.58%）远高于测试集（93.17%），这是决策树集成方法的典型特征——单棵树的复杂度叠加可能拟合了训练集中的噪声。尽管如此，XGBoost 仍取得了最高的测试集精度，说明这种程度的\"过拟合\"在可接受范围内。"),
);

// ===== CHAPTER 4: 系统展示 =====
children.push(
  new Paragraph({ pageBreakBefore: true }),
  heading1("第四章  系统展示"),

  heading2("4.1  工程架构"),
  bodyTextIndent("本项目按照标准机器学习工程项目的文件夹架构组织，确保代码的模块化、可维护性和可复现性。各个模块职责明确、接口清晰："),

  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [3000, 6360],
    rows: [
      new TableRow({ children: [headerCell("模块文件", 3000), headerCell("功能说明", 6360)] }),
      new TableRow({ children: [dataCell("src/data_loader.py", 3000, true), dataCell("数据加载：读取CSV，处理编码和特殊值", 6360, true)] }),
      new TableRow({ children: [dataCell("src/preprocessor.py", 3000), dataCell("预处理流水线：标签清洗、缺失值填充、标准化", 6360)] }),
      new TableRow({ children: [dataCell("src/visualizer.py", 3000, true), dataCell("可视化：生成所有分析图表(matplotlib+seaborn)", 6360, true)] }),
      new TableRow({ children: [dataCell("src/noise_generator.py", 3000), dataCell("噪声生成：高斯噪声、标签噪声、缺失值注入", 6360)] }),
      new TableRow({ children: [dataCell("src/trainer.py", 3000, true), dataCell("训练调度：统一训练接口、模型持久化", 6360, true)] }),
      new TableRow({ children: [dataCell("src/evaluator.py", 3000), dataCell("评估系统：5维度评估（精度/曲线/速度/鲁棒性/过拟合）", 6360)] }),
      new TableRow({ children: [dataCell("src/algorithms/*.py", 3000, true), dataCell("算法模块：KNN/LR/SVM/ANN(手写)/XGBoost(课外)", 6360, true)] }),
      new TableRow({ children: [dataCell("main.py", 3000), dataCell("CLI统一入口：5个子命令控制全流程", 6360)] }),
    ]
  }),
  new Paragraph({ spacing: { after: 60 } }),
  bodyText("表4-1 工程模块说明"),

  heading2("4.2  命令行接口 (CLI)"),
  bodyTextIndent("本项目设计了统一的命令行接口，通过 main.py 实现对全流程的控制。所有算法运行阶段不显示 UI 界面，仅在命令行输出进度和结果，符合课程要求。"),

  bodyText("命令行接口如下："),
  bodyText("  python main.py analyze        # 运行数据分析，生成4张图表"),
  bodyText("  python main.py preprocess     # 运行数据预处理流水线"),
  bodyText("  python main.py train          # 训练所有5种算法"),
  bodyText("  python main.py evaluate       # 多维度评估，生成6张对比图表"),
  bodyText("  python main.py full-pipeline  # 一键运行完整流程"),

  bodyTextIndent("每个命令独立可运行，输出保存到 outputs/ 目录下的对应子目录（figures/、models/、results/），便于查看和引用。"),

  heading2("4.3  GitHub展示"),
  bodyTextIndent("项目已上传至 GitHub，方便教师在线查看完整代码、文档和实验结果。README.md 文件按照课程评分要求，包含以下内容："),

  bodyText("  (1) 数据集描述与数据污染说明"),
  bodyText("  (2) 数据处理方法列表（标签清洗、缺失值填充、特征工程）"),
  bodyText("  (3) 实现的算法清单（含课内/课外标记和手写ANN说明）"),
  bodyText("  (4) 所有算法的精度对比汇总表"),
  bodyText("  (5) 工程架构目录树"),
  bodyText("  (6) 快速开始指南（环境安装+命令行运行）"),
  bodyText("  (7) 课程总结"),

  bodyTextIndent("GitHub 仓库展示网页链接将在正式提交时补充到论文中。"),
);

// ===== CHAPTER 5: 课程总结 =====
children.push(
  new Paragraph({ pageBreakBefore: true }),
  heading1("第五章  课程总结"),

  heading2("5.1  所学内容"),
  bodyTextIndent("通过本学期的机器学习课程学习，我系统地掌握了机器学习的基本理论和实践技能，具体包括以下几个方面："),

  bodyTextIndent("（1）机器学习基础知识：理解了机器学习的基本概念（监督学习、无监督学习、分类、回归）、模型的偏差-方差权衡（过拟合与欠拟合）、交叉验证等核心理论。"),
  bodyTextIndent("（2）经典分类算法：学习并实践了 KNN（K近邻）、Logistic Regression（逻辑回归）、SVM（支持向量机）等传统机器学习算法，理解了它们的数学模型、优化目标和适用场景。"),
  bodyTextIndent("（3）神经网络与深度学习：重点学习了人工神经网络（ANN）的前向传播和反向传播算法，从数学推导到代码实现完整地理解了神经网络的训练过程。同时在课堂上了解了 CNN（卷积神经网络）、VGG 等更深层的网络结构，以及 HOG 等传统特征提取方法。"),
  bodyTextIndent("（4）数据预处理与特征工程：掌握了缺失值处理、标签清洗、特征标准化、异常值检测等数据清洗技巧，理解了\"垃圾进垃圾出\"的道理——数据质量直接决定模型上限。"),
  bodyTextIndent("（5）模型评估方法论：学会了从多个维度（精度、收敛速度、推理效率、鲁棒性、过拟合程度）全面评估和对比不同算法，建立了科学的实验设计思维。"),
  bodyTextIndent("（6）工程化能力：通过本次期末大作业，实践了将机器学习流程模块化、工程化的全过程——从数据加载、预处理、模型训练到评估分析，构建了可复用的命令行工具。"),

  heading2("5.2  课程评价与建议"),
  bodyTextIndent("课程整体评价：本课程内容充实、理论与实践结合紧密，是一门前沿且实用的课程。教师授课逻辑清晰，从基础的 KNN、逻辑回归出发，逐步深入到 SVM、ANN、CNN，使我在循序渐近的学习过程中建立了扎实的机器学习知识体系。"),

  bodyTextIndent("课程的亮点在于以下几个方面："),
  bodyTextIndent("（1）理论与实践并重：每学完一种算法都有对应的实验和作业，帮助巩固理论理解。特别是要求手写 ANN 的前向和反向传播，这种\"造轮子\"的练习对理解神经网络的本质非常有价值。"),
  bodyTextIndent("（2）期末大作业设计合理：要求涵盖数据分析、数据处理、多算法实验、系统展示和课程总结的全流程，全面检验了学生的综合能力。同时鼓励学生自学一种课外算法（如本实验中的 XGBoost），培养了自主学习能力。"),
  bodyTextIndent("（3）脏数据版本的引入：教师提供的 Dirty Dataset 包含多种类型的污染数据，这一设计非常贴近实际工作场景——真实世界的数据从来不是干净的，学会处理脏数据是机器学习工程师的必备技能。"),

  bodyTextIndent("建议与改进方向："),
  bodyTextIndent("（1）可以增加一些关于模型解释性（如 SHAP、LIME）的内容，帮助理解模型的决策依据。"),
  bodyTextIndent("（2）建议在课程中增加一些模型部署相关的内容（如模型导出、API 服务化），帮助弥合\"实验\"到\"生产\"的鸿沟。"),
  bodyTextIndent("（3）可以考虑引入更多真实工业场景的数据集，让学生接触更贴近实际的问题类型（如极度不均衡分类、多标签分类等）。"),

  heading2("5.3  结语"),
  bodyTextIndent("通过本学期的学习和本次期末大作业的实践，我不仅掌握了机器学习的核心理论知识，更重要的是建立了从数据到模型的完整工程化思维。感谢教师的悉心指导和课程团队的辛勤付出。这套方法论将为我未来的学习和工作奠定坚实的基础。"),
);

// ===== 参考文献 =====
children.push(
  new Paragraph({ pageBreakBefore: true }),
  heading1("参考文献"),
  bodyText("[1] Koklu, M., & Ozkan, I. A. (2020). Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques. Computers and Electronics in Agriculture, 174, 105507."),
  bodyText("[2] UCI Machine Learning Repository. Dry Bean Dataset. https://archive.ics.uci.edu/dataset/602/dry+bean+dataset"),
  bodyText("[3] Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016."),
  bodyText("[4] Scikit-learn: Machine Learning in Python. https://scikit-learn.org/"),
  bodyText("[5] He, K., et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. ICCV 2015."),
  bodyText("[6] Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer."),
  bodyText("[7] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press."),
);

// ===== BUILD DOCUMENT =====
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Microsoft YaHei", size: 22 } }
    },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Microsoft YaHei", color: "1F4E79" },
        paragraph: { spacing: { before: 360, after: 240 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Microsoft YaHei", color: "2E75B6" },
        paragraph: { spacing: { before: 240, after: 180 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Microsoft YaHei", color: "333333" },
        paragraph: { spacing: { before: 180, after: 120 }, outlineLevel: 2 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },  // A4
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "机器学习期末大作业 | 基于Dry Bean Dataset的全流程ML工程", font: "Microsoft YaHei", size: 16, color: "999999", italics: true })]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "- ", font: "Microsoft YaHei", size: 16 }), new TextRun({ children: [PageNumber.CURRENT], font: "Microsoft YaHei", size: 16 }), new TextRun({ text: " -", font: "Microsoft YaHei", size: 16 })]
        })]
      })
    },
    children
  }]
});

const outPath = `${PROJECT}/期末论文.docx`;
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outPath, buffer);
  console.log(`Paper saved to: ${outPath}`);
  console.log(`Size: ${(buffer.length / 1024).toFixed(0)} KB`);
});
