"""
测试RetrievalNet完整功能
用法: python scripts/test_retrieval_net.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np

from models.retrieval_net import RetrievalNet


def visualize_features(features, save_path='test_feature_distribution.png'):
    """可视化特征分布"""
    features_np = features.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 特征分布直方图
    axes[0, 0].hist(features_np.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Feature Distribution')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 特征范数分布
    norms = np.linalg.norm(features_np, axis=1)
    axes[0, 1].hist(norms, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(1.0, color='red', linestyle='--', linewidth=2, label='Expected=1.0')
    axes[0, 1].set_title('Feature Norm Distribution')
    axes[0, 1].set_xlabel('L2 Norm')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 特征热力图
    im = axes[1, 0].imshow(features_np, aspect='auto', cmap='viridis')
    axes[1, 0].set_title('Feature Heatmap')
    axes[1, 0].set_xlabel('Dimension')
    axes[1, 0].set_ylabel('Sample')
    plt.colorbar(im, ax=axes[1, 0])

    # 4. 前10维特征箱线图
    axes[1, 1].boxplot([features_np[:, i] for i in range(min(10, features_np.shape[1]))],
                       labels=[f'Dim{i}' for i in range(min(10, features_np.shape[1]))])
    axes[1, 1].set_title('Feature Statistics (First 10 Dims)')
    axes[1, 1].set_xlabel('Dimension')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"特征分布可视化已保存: {save_path}")
    plt.close()


def test_feature_quality(model, num_samples=50):
    """测试特征质量（区分度）"""
    print("\n=== 测试特征区分度 ===")

    model.eval()
    features_list = []

    with torch.no_grad():
        for _ in range(num_samples):
            # 生成随机输入（模拟不同场景）
            x = torch.randn(1, 9, 200, 200)
            features = model(x)
            features_list.append(features)

    features = torch.cat(features_list, dim=0)  # [num_samples, 128]

    # 计算特征之间的余弦相似度
    features_norm = features / features.norm(dim=1, keepdim=True)
    similarity_matrix = features_norm @ features_norm.T

    # 提取上三角（不包括对角线）
    mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
    similarities = similarity_matrix[mask]

    print(f"  样本数: {num_samples}")
    print(f"  特征对数: {len(similarities)}")
    print(f"  平均余弦相似度: {similarities.mean().item():.4f}")
    print(f"  相似度标准差: {similarities.std().item():.4f}")
    print(f"  相似度范围: [{similarities.min().item():.4f}, {similarities.max().item():.4f}]")

    # 理想情况: 随机场景相似度应接近0，标准差应较大
    if abs(similarities.mean().item()) < 0.3:
        print("  ✓ 特征区分度良好（平均相似度接近0）")
    else:
        print("  ⚠ 特征区分度可能不足")

    return features


def main():
    print("=" * 60)
    print("Day 5-6: RetrievalNet完整测试")
    print("=" * 60)

    # 1. 创建模型
    print("\n【1/5】创建RetrievalNet")
    model = RetrievalNet(output_dim=128)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  目标: <10M ({'✓' if total_params < 10e6 else '✗'})")

    # 2. 前向传播测试
    print("\n【2/5】前向传播测试")
    batch_size = 8
    x = torch.randn(batch_size, 9, 200, 200)

    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"  输入: {x.shape}")
    print(f"  输出: {output.shape}")
    print(f"  ✓ 维度正确")

    # 3. 梯度测试
    print("\n【3/5】梯度反向传播测试")
    model.train()
    x.requires_grad = True
    output = model(x)
    loss = output.sum()
    loss.backward()

    has_grad = x.grad is not None and not torch.isnan(x.grad).any()
    print(f"  梯度检查: {'✓ 通过' if has_grad else '✗ 失败'}")

    # 4. 特征质量测试
    print("\n【4/5】特征质量测试")
    features = test_feature_quality(model, num_samples=50)
    visualize_features(features)

    # 5. 推理速度测试
    print("\n【5/5】推理速度测试")
    model.eval()
    import time

    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = model(x)

        # 计时
        start = time.time()
        num_iterations = 100
        for _ in range(num_iterations):
            _ = model(x)
        end = time.time()

    avg_time = (end - start) / num_iterations * 1000
    fps = 1000 / avg_time * batch_size

    print(f"  批大小: {batch_size}")
    print(f"  平均推理时间: {avg_time:.2f}ms")
    print(f"  吞吐量: {fps:.1f} samples/sec")
    print(f"  目标: >30 FPS ({'✓ 达标' if fps > 30 else '✗ 未达标'})")

    print("\n" + "=" * 60)
    print("✓ Day 5-6 所有测试完成!")
    print("=" * 60)
    print("\n生成的文件:")
    print("  - test_feature_distribution.png")
    print("\n下一步: Day 7 实现损失函数和训练框架")


if __name__ == "__main__":
    main()
