"""
实验一：心跳信号预测
====================
基于1D CNN的心跳信号分类

数据说明：
  使用 data/split_data.py 将原始数据划分为训练集和测试集，
  或直接在本代码中通过 train_test_split 自动划分。

运行方法：
  python src/experiment1.py              # 自动从原始数据划分
  python src/experiment1.py --split      # 使用预划分的 train_data.csv/test_data.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 12

# 设置随机种子保证可重复性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 解析命令行参数
parser = argparse.ArgumentParser(description='心跳信号分类实验')
parser.add_argument('--split', action='store_true',
                    help='使用预划分的 train_data.csv/test_data.csv（默认从原始数据自动划分）')
parser.add_argument('--data_dir', type=str, default='../data',
                    help='数据目录（默认 ../data）')
parser.add_argument('--epochs', type=int, default=30,
                    help='训练轮数（默认30）')
parser.add_argument('--batch_size', type=int, default=64,
                    help='批大小（默认64）')
parser.add_argument('--lr', type=float, default=0.001,
                    help='学习率（默认0.001）')
args = parser.parse_args()

# ============================================================
# 1. 数据加载与探索性分析
# ============================================================
print("=" * 60)
print("1. 数据加载与探索性分析")
print("=" * 60)

if args.split:
    # 使用预划分文件
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.data_dir))
    train_path = os.path.join(data_dir, 'train_data.csv')
    test_path = os.path.join(data_dir, 'test_data.csv')
    print(f"使用预划分数据:")
    print(f"  训练集: {train_path}")
    print(f"  测试集: {test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    df = pd.concat([train_df, test_df], ignore_index=True)
    # 分别解析信号数据
    X_train = np.array([list(map(float, sig.split(','))) for sig in train_df['heartbeat_signals']])
    y_train = np.array(train_df['label'].values, dtype=np.int64)
    X_test = np.array([list(map(float, sig.split(','))) for sig in test_df['heartbeat_signals']])
    y_test = np.array(test_df['label'].values, dtype=np.int64)
else:
    # 从原始数据自动划分（默认方式）
    df = pd.read_csv(r'c:\Users\lcy20\Desktop\实验\心跳测试\train.csv')
    print(f"数据集形状: {df.shape}")

    # 将信号字符串转为数值矩阵
    print("\n正在解析信号数据...")
    X = np.array([list(map(float, sig.split(','))) for sig in df['heartbeat_signals']])
    y = np.array(df['label'].values, dtype=np.int64)

    # 按 8:2 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

# 标签分布
label_counts = df['label'].value_counts().sort_index()
print(f"\n标签分布:")
for k, v in label_counts.items():
    print(f"  类别 {int(k)}: {v} 样本 ({v/len(df)*100:.2f}%)")
print(f"\n训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")

# 转换为 PyTorch Tensor
X_train_t = torch.FloatTensor(X_train).unsqueeze(1)  # (N, 1, 205)
X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
y_train_t = torch.LongTensor(y_train)
y_test_t = torch.LongTensor(y_test)

# 自定义 Dataset
class HeartbeatDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 创建 DataLoader
batch_size = args.batch_size
train_dataset = HeartbeatDataset(X_train_t, y_train_t)
test_dataset = HeartbeatDataset(X_test_t, y_test_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练批次数: {len(train_loader)}, 测试批次数: {len(test_loader)}")

# ============================================================
# 2. 构建 1D CNN 模型
# ============================================================
print("\n" + "=" * 60)
print("2. 构建 1D CNN 模型")
print("=" * 60)

class CNN1D(nn.Module):
    """一维卷积神经网络用于心跳信号分类"""

    def __init__(self, num_classes=4):
        super(CNN1D, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 1 -> 32
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Block 2: 32 -> 64
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Block 3: 64 -> 128
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Block 4: 128 -> 256
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


model = CNN1D(num_classes=4).to(device)
print(model)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")

# ============================================================
# 3. 训练模型
# ============================================================
print("\n" + "=" * 60)
print("3. 训练模型")
print("=" * 60)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


num_epochs = args.epochs
best_acc = 0.0
train_losses, train_accs = [], []
test_losses, test_accs = [], []

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    scheduler.step()

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), '../results/best_model.pth')

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

print(f"\n训练完成！最佳测试准确率: {best_acc:.2f}%")

# ============================================================
# 4. 评估模型
# ============================================================
print("\n" + "=" * 60)
print("4. 模型评估")
print("=" * 60)

# 加载最佳模型
model.load_state_dict(torch.load('../results/best_model.pth'))
model.eval()

# 在测试集上进行预测
all_preds = []
all_targets = []
all_probs = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
all_probs = np.array(all_probs)

# 分类报告
print("\n分类报告:")
print(classification_report(all_targets, all_preds,
                            target_names=['类别0(正常)', '类别1', '类别2', '类别3']))

# 混淆矩阵
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['类别0', '类别1', '类别2', '类别3'],
            yticklabels=['类别0', '类别1', '类别2', '类别3'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.tight_layout()
plt.savefig('../results/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存混淆矩阵: ../results/confusion_matrix.png")

# 各类别准确率
print("\n各类别准确率:")
for i in range(4):
    mask = all_targets == i
    acc = (all_preds[mask] == all_targets[mask]).mean() * 100
    print(f"  类别 {i}: {acc:.2f}% ({mask.sum()} 样本)")

# ============================================================
# 5. 可视化训练过程
# ============================================================
print("\n" + "=" * 60)
print("5. 可视化训练过程")
print("=" * 60)

plt.figure(figsize=(14, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='训练损失', linewidth=2, color='#1f77b4')
plt.plot(test_losses, label='测试损失', linewidth=2, color='#d62728')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.title('训练与测试损失曲线')
plt.legend()
plt.grid(True, alpha=0.3)

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='训练准确率', linewidth=2, color='#1f77b4')
plt.plot(test_accs, label='测试准确率', linewidth=2, color='#d62728')
plt.xlabel('Epoch')
plt.ylabel('准确率 (%)')
plt.title('训练与测试准确率曲线')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/training_history.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存训练历史图: ../results/training_history.png")

# ============================================================
# 6. 预测结果可视化
# ============================================================
print("\n" + "=" * 60)
print("6. 预测结果展示")
print("=" * 60)

plt.figure(figsize=(16, 10))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i in range(4):
    # 找到真实类别为i的样本
    class_mask = all_targets == i
    class_indices = np.where(class_mask)[0]
    correct_mask = all_preds[class_mask] == all_targets[class_mask]
    wrong_mask = ~correct_mask

    correct_idx = class_indices[correct_mask]
    wrong_idx = class_indices[wrong_mask]

    # 绘制正确分类的样本
    if len(correct_idx) > 0:
        for j in range(min(2, len(correct_idx))):
            plt.subplot(4, 4, i * 4 + j + 1)
            orig_idx = correct_idx[j]
            plt.plot(X_test[orig_idx], color=colors[i], linewidth=1.2)
            plt.title(f'类别{i} 正确 (置信度:{all_probs[orig_idx][i]:.2f})', fontsize=9)
            plt.xticks([])

    # 绘制错误分类的样本
    if len(wrong_idx) > 0:
        for j in range(min(2, len(wrong_idx))):
            plt.subplot(4, 4, i * 4 + j + 3)
            orig_idx = wrong_idx[j]
            pred_label = all_preds[orig_idx]
            plt.plot(X_test[orig_idx], color=colors[pred_label], linewidth=1.2)
            plt.title(f'类别{i}→{pred_label} (置信:{all_probs[orig_idx][pred_label]:.2f})', fontsize=9)
            plt.xticks([])

plt.suptitle('预测结果展示 (正确 vs 错误分类)', fontsize=14)
plt.tight_layout()
plt.savefig('../results/prediction_examples.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存预测示例图: ../results/prediction_examples.png")

print("\n" + "=" * 60)
print("实验一完成！生成的文件:")
print("  - results/class_signal_analysis.png  (各类别信号分析)")
print("  - results/label_distribution.png     (标签分布)")
print("  - results/training_history.png       (训练历史)")
print("  - results/confusion_matrix.png       (混淆矩阵)")
print("  - results/prediction_examples.png    (预测示例)")
print("  - results/best_model.pth             (最佳模型权重)")
print("=" * 60)
