"""
BEV缓存数据分析
用法: python scripts/analyze_cache.py
"""

import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

cache_root = "/home/wzj/pan1/contour_deep/data/Chilean_BEV_Cache/"


def analyze_split(split: str):
    """分析训练集或测试集"""
    cache_dir = os.path.join(cache_root, split)
    cache_files = list(Path(cache_dir).glob("*.npz"))

    print(f"\n分析 {split} 集: {len(cache_files)} 个文件")

    # 统计指标
    layer_occupancy = np.zeros(8)  # 每层平均占用像素
    vcd_values = []  # VCD值分布

    # 随机采样1000个文件（加速分析）
    import random
    sample_files = random.sample(cache_files, min(1000, len(cache_files)))

    for cache_file in tqdm(sample_files, desc="Analyzing"):
        data = np.load(cache_file)
        bev_layers = data['bev_layers']
        vcd = data['vcd']

        # 统计每层占用
        for i in range(8):
            layer_occupancy[i] += np.sum(bev_layers[i] > 0)

        # 统计VCD
        vcd_values.extend(vcd.flatten().tolist())

    # 计算平均值
    layer_occupancy /= len(sample_files)

    # 结果
    stats = {
        'split': split,
        'total_files': len(cache_files),
        'sampled_files': len(sample_files),
        'layer_occupancy': layer_occupancy.tolist(),
        'vcd_distribution': dict(Counter(vcd_values)),
        'avg_occupied_pixels': layer_occupancy.mean(),
        'total_occupied_pixels': layer_occupancy.sum()
    }

    return stats


def visualize_stats(train_stats, test_stats):
    """可视化统计结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 每层占用像素数（训练集）
    ax = axes[0, 0]
    ax.bar(range(8), train_stats['layer_occupancy'])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Avg Occupied Pixels')
    ax.set_title('Train: Layer Occupancy')
    ax.grid(True, alpha=0.3)

    # 2. 每层占用像素数（测试集）
    ax = axes[0, 1]
    ax.bar(range(8), test_stats['layer_occupancy'])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Avg Occupied Pixels')
    ax.set_title('Test: Layer Occupancy')
    ax.grid(True, alpha=0.3)

    # 3. VCD分布（训练集）
    ax = axes[1, 0]
    vcd_dist = train_stats['vcd_distribution']
    ax.bar(vcd_dist.keys(), vcd_dist.values())
    ax.set_xlabel('VCD Value')
    ax.set_ylabel('Count')
    ax.set_title('Train: VCD Distribution')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 4. VCD分布（测试集）
    ax = axes[1, 1]
    vcd_dist = test_stats['vcd_distribution']
    ax.bar(vcd_dist.keys(), vcd_dist.values())
    ax.set_xlabel('VCD Value')
    ax.set_ylabel('Count')
    ax.set_title('Test: VCD Distribution')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cache_results/dataset_stats.png', dpi=150)
    print("\n可视化已保存: cache_results/dataset_stats.png")


def main():
    os.makedirs('cache_results', exist_ok=True)

    # 分析训练集和测试集
    train_stats = analyze_split('train')
    test_stats = analyze_split('test')

    # 打印统计结果
    print("\n" + "=" * 60)
    print("统计结果:")
    print("=" * 60)
    print(f"\n训练集:")
    print(f"  文件数: {train_stats['total_files']}")
    print(f"  平均每层占用: {train_stats['avg_occupied_pixels']:.1f} pixels")
    print(f"  总占用: {train_stats['total_occupied_pixels']:.1f} pixels")
    print(f"  Layer占用: {[f'{x:.0f}' for x in train_stats['layer_occupancy']]}")

    print(f"\n测试集:")
    print(f"  文件数: {test_stats['total_files']}")
    print(f"  平均每层占用: {test_stats['avg_occupied_pixels']:.1f} pixels")
    print(f"  总占用: {test_stats['total_occupied_pixels']:.1f} pixels")
    print(f"  Layer占用: {[f'{x:.0f}' for x in test_stats['layer_occupancy']]}")

    # 保存为JSON
    with open('cache_results/dataset_stats.json', 'w') as f:
        json.dump({
            'train': train_stats,
            'test': test_stats
        }, f, indent=2)
    print("\n统计结果已保存: cache_results/dataset_stats.json")

    # 可视化
    visualize_stats(train_stats, test_stats)


if __name__ == "__main__":
    main()
