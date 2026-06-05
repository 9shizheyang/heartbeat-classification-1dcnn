"""
数据划分脚本
============
将原始 train.csv 按 80/20 分层抽样划分为训练集和测试集。
确保划分结果与实验代码完全一致（random_state=42）。

用法:
    python split_data.py                        # 划分数据
    python split_data.py --input <文件路径>       # 指定原始数据
    python split_data.py --output <输出目录>      # 指定输出目录
"""

import os
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(input_path: str, output_dir: str, test_size: float = 0.2,
               random_state: int = 42):
    """
    将原始数据按分层抽样划分为训练集和测试集。

    Parameters
    ----------
    input_path : str
        原始 CSV 文件路径
    output_dir : str
        输出目录路径
    test_size : float
        测试集比例，默认 0.2
    random_state : int
        随机种子，默认 42（确保可重复性）
    """
    print(f"读取数据: {input_path}")
    df = pd.read_csv(input_path)
    print(f"总样本数: {len(df)}")
    print(f"原始标签分布:")
    orig_dist = df['label'].value_counts().sort_index()
    for k, v in orig_dist.items():
        print(f"  类别 {int(k)}: {v} 条 ({v / len(df) * 100:.2f}%)")

    # 分层抽样划分
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )

    print(f"\n划分结果:")
    print(f"  训练集: {len(train_df)} 条 ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"  测试集: {len(test_df)} 条 ({len(test_df) / len(df) * 100:.1f}%)")

    # 检查分布一致性
    print(f"\n训练集标签分布:")
    train_dist = train_df['label'].value_counts().sort_index()
    for k, v in train_dist.items():
        print(f"  类别 {int(k)}: {v} 条 ({v / len(train_df) * 100:.2f}%)")
    print(f"\n测试集标签分布:")
    test_dist = test_df['label'].value_counts().sort_index()
    for k, v in test_dist.items():
        print(f"  类别 {int(k)}: {v} 条 ({v / len(test_df) * 100:.2f}%)")

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n✅ 已保存:")
    print(f"  训练集: {train_path} ({len(train_df)} 条)")
    print(f"  测试集: {test_path} ({len(test_df)} 条)")

    return train_df, test_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='心跳信号数据划分')
    parser.add_argument('--input', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'train.csv'),
                        help='原始数据文件路径')
    parser.add_argument('--output', type=str,
                        default=os.path.dirname(__file__),
                        help='输出目录路径')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例（默认0.2）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认42）')
    args = parser.parse_args()

    split_data(args.input, args.output, args.test_size, args.seed)
