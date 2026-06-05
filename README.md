# 心跳信号分类实验（基于1D CNN）

## 实验概述

本实验利用**一维卷积神经网络（1D CNN）** 对心电图（ECG）心跳信号进行分类，实现心电信号的自动识别。实验基于PyTorch框架，在心电信号四分类任务上取得了 **98.99%** 的测试准确率。

## 项目结构

```
├── README.md                    # 本文件
├── requirements.txt             # Python依赖包
├── .gitignore                   # Git忽略配置
├── 实验报告_心跳信号分类.docx    # 实验报告
├── data/
│   ├── split_data.py            # 数据划分脚本
│   ├── train_data.csv           # 训练集 (80,000条)
│   └── test_data.csv            # 测试集 (20,000条)
├── src/
│   └── experiment1.py           # 实验主代码
└── results/
    ├── class_signal_analysis.png    # 各类别信号分析图
    ├── label_distribution.png       # 标签分布图
    ├── training_history.png         # 训练历史曲线
    ├── confusion_matrix.png         # 混淆矩阵
    ├── prediction_examples.png      # 预测示例图
    └── best_model.pth               # 最佳模型权重
```

## 数据集

- **总样本数**: 100,000 条心跳信号记录
- **信号长度**: 205 个时间点/条
- **类别数**: 4 类（0: 正常, 1-3: 异常类型）
- **划分方式**: 80% 训练集 / 20% 测试集（分层抽样）

### 标签分布

| 类别 | 含义 | 样本数 | 占比 |
|------|------|--------|------|
| 0 | 正常心跳 | 64,327 | 64.33% |
| 1 | 异常类型1 | 3,562 | 3.56% |
| 2 | 异常类型2 | 14,199 | 14.20% |
| 3 | 异常类型3 | 17,912 | 17.91% |

## 模型架构

**CNN1D** -一维卷积神经网络（总参数量: 168,196）

```
输入 (1×205)
  └─ Block 1: Conv1d(1→32, k=7) + BN + ReLU + MaxPool
  └─ Block 2: Conv1d(32→64, k=5) + BN + ReLU + MaxPool
  └─ Block 3: Conv1d(64→128, k=3) + BN + ReLU + MaxPool
  └─ Block 4: Conv1d(128→256, k=3) + BN + ReLU + AdaptiveMaxPool
  └─ Classifier: Dropout(0.5) → Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→4)
输出 (4类概率)
```

## 运行方法

### 环境配置

```bash
pip install -r requirements.txt
```

### 方式一：自动划分数据后训练（需要原始 train.csv）

```bash
python src/experiment1.py
```

### 方式二：使用预划分的 train_data.csv / test_data.csv

```bash
# 先划分数据
python data/split_data.py --input path/to/train.csv --output data/

# 再训练
python src/experiment1.py --split
```

### 自定义训练参数

```bash
python src/experiment1.py --epochs 50 --batch_size 128 --lr 0.0005
```

## 实验结果

| 类别 | 精确率 | 召回率 | F1分数 | 准确率 |
|------|--------|--------|--------|--------|
| 类别0（正常） | 0.99 | 1.00 | 0.99 | 99.61% |
| 类别1 | 0.95 | 0.86 | 0.90 | 85.67% |
| 类别2 | 0.99 | 0.99 | 0.99 | 98.56% |
| 类别3 | 1.00 | 1.00 | 1.00 | 99.75% |
| **加权平均** | **0.99** | **0.99** | **0.99** | **98.99%** |

## 依赖

- Python 3.7+
- PyTorch
- NumPy, Pandas, scikit-learn
- Matplotlib, Seaborn
- python-docx
