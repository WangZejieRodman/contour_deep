"""
测试数据集的专用脚本
用法: python scripts/test_dataset.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.dataset_retrieval import RetrievalDataset, create_dataloader


def test_loading_speed(dataloader, num_epochs=1):
    """测试加载速度"""
    print("\n=== 测试加载速度 ===")

    total_samples = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
            total_samples += batch['anchor'].shape[0]

            # 模拟GPU传输
            if torch.cuda.is_available():
                batch['anchor'].cuda()
                batch['positive'].cuda()
                batch['negatives'].cuda()

    elapsed = time.time() - start_time
    samples_per_sec = total_samples / elapsed

    print(f"\n总样本数: {total_samples}")
    print(f"总耗时: {elapsed:.2f}秒")
    print(f"加载速度: {samples_per_sec:.1f} samples/sec")

    return samples_per_sec


def visualize_batch(batch, save_path='test_batch_visualization.png'):
    """可视化一个batch"""
    print("\n=== 可视化batch ===")

    batch_size = batch['anchor'].shape[0]
    num_samples = min(3, batch_size)  # 显示前3个样本

    fig, axes = plt.subplots(num_samples, 9, figsize=(18, num_samples * 2))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Anchor (3层BEV)
        for j in range(3):
            axes[i, j].imshow(batch['anchor'][i, j].cpu().numpy(), cmap='viridis')
            axes[i, j].set_title(f'A{i} L{j}', fontsize=8)
            axes[i, j].axis('off')

        # Positive (3层BEV)
        for j in range(3):
            axes[i, j + 3].imshow(batch['positive'][i, j].cpu().numpy(), cmap='viridis')
            axes[i, j + 3].set_title(f'P{i} L{j}', fontsize=8)
            axes[i, j + 3].axis('off')

        # Negative[0] (3层BEV)
        for j in range(3):
            axes[i, j + 6].imshow(batch['negatives'][i, 0, j].cpu().numpy(), cmap='viridis')
            axes[i, j + 6].set_title(f'N{i} L{j}', fontsize=8)
            axes[i, j + 6].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"可视化已保存: {save_path}")
    plt.close()


def test_augmentation(dataset):
    """测试数据增强效果"""
    print("\n=== 测试数据增强 ===")

    # 获取同一个样本多次（观察增强效果）
    sample_idx = 0
    samples = [dataset[sample_idx] for _ in range(5)]

    fig, axes = plt.subplots(5, 3, figsize=(10, 15))

    for i, sample in enumerate(samples):
        for j in range(3):
            axes[i, j].imshow(sample['anchor'][j].cpu().numpy(), cmap='viridis')
            axes[i, j].set_title(f'Aug {i + 1} - Layer {j}', fontsize=8)
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig('test_augmentation_effect.png', dpi=150)
    print("增强效果已保存: test_augmentation_effect.png")
    plt.close()


def check_data_distribution(dataloader):
    """检查数据分布"""
    print("\n=== 检查数据分布 ===")

    all_means = []
    all_stds = []

    for batch in tqdm(dataloader, desc="Computing statistics"):
        anchors = batch['anchor'].cpu().numpy()
        all_means.append(anchors.mean())
        all_stds.append(anchors.std())

    mean = np.mean(all_means)
    std = np.mean(all_stds)

    print(f"数据集均值: {mean:.4f}")
    print(f"数据集标准差: {std:.4f}")

    return mean, std


def main():
    # 加载配置
    config_path = "/home/wzj/pan1/contour_deep/configs/config_base.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("Day 3: RetrievalDataset 完整测试")
    print("=" * 60)

    # 1. 创建训练数据集
    print("\n【1/6】创建训练数据集")
    train_dataset = RetrievalDataset(
        queries_pickle="/home/wzj/pan1/contour_deep/data/test_queries_chilean_period.pickle",
        cache_root="/home/wzj/pan1/contour_deep/data/Chilean_BEV_Cache/",
        split='train',
        num_negatives=10,
        augmentation_config=config['augmentation'],
        resolution=config['bev']['resolution'],
        use_cache=True,
        max_cache_size=500
    )

    # 2. 创建训练DataLoader
    print("\n【2/6】创建训练DataLoader")
    train_loader = create_dataloader(
        train_dataset,
        batch_size=8,
        num_workers=4,
        shuffle=True
    )

    # 3. 测试加载速度
    print("\n【3/6】测试加载速度")
    speed = test_loading_speed(train_loader, num_epochs=1)

    if speed > 50:
        print("✓ 加载速度达标!")
    else:
        print("⚠ 加载速度不足，建议:")
        print("  - 增加num_workers")
        print("  - 增加max_cache_size")
        print("  - 检查磁盘IO性能")

    # 4. 可视化batch
    print("\n【4/6】可视化batch")
    batch = next(iter(train_loader))
    visualize_batch(batch)

    # 5. 测试数据增强
    print("\n【5/6】测试数据增强")
    test_augmentation(train_dataset)

    # 6. 检查数据分布
    print("\n【6/6】检查数据分布")
    mean, std = check_data_distribution(train_loader)

    # 7. 创建测试数据集
    print("\n【额外】创建测试数据集")
    test_dataset = RetrievalDataset(
        queries_pickle="/home/wzj/pan1/contour_deep/data/test_queries_chilean_period.pickle",
        cache_root="/home/wzj/pan1/contour_deep/data/Chilean_BEV_Cache/",
        split='test',
        num_negatives=10,
        augmentation_config=None,
        resolution=config['bev']['resolution'],
        use_cache=True
    )

    test_loader = create_dataloader(
        test_dataset,
        batch_size=8,
        num_workers=4,
        shuffle=False
    )

    print(f"测试集大小: {len(test_dataset)}")
    print(f"测试集batches: {len(test_loader)}")

    print("\n" + "=" * 60)
    print("✓ Day 3 所有测试完成!")
    print("=" * 60)
    print("\n生成的文件:")
    print("  - test_batch_visualization.png")
    print("  - test_augmentation_effect.png")
    print("\n下一步: Day 4-6 实现方向1网络架构")


if __name__ == "__main__":
    main()
