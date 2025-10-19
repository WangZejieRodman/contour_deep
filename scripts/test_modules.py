"""
测试Day 4模块
用法: python scripts/test_modules.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.modules import MultiScaleConv, SpatialAttention, CrossLayerAttention


def test_multiscale_conv():
    print("\n=== 测试 MultiScaleConv ===")

    model = MultiScaleConv(in_channels=9, out_channels=128)# 创建模块

    x = torch.randn(4, 9, 200, 200)# 输入: 一个batch的BEV数据，[batch=4, channels=9(8层BEV+1层VCD), H=200, W=200]

    out = model(x)# 前向传播 预期输出: [4, 128, 200, 200]

    print(f"✓ Input: {x.shape}")
    print(f"✓ Output: {out.shape}")
    print(f"✓ Expected: [4, 128, 200, 200]")

    assert out.shape == (4, 128, 200, 200)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Parameters: {num_params:,}")

    return model


def test_spatial_attention():
    print("\n=== 测试 SpatialAttention ===")

    model = SpatialAttention()
    x = torch.randn(4, 128, 200, 200)

    out = model(x)

    print(f"✓ Input: {x.shape}")
    print(f"✓ Output: {out.shape}")

    assert out.shape == x.shape

    # 检查注意力效果
    print(f"✓ Input mean: {x.mean().item():.4f}")
    print(f"✓ Output mean: {out.mean().item():.4f}")

    return model


def test_cross_layer_attention():
    print("\n=== 测试 CrossLayerAttention ===")

    model = CrossLayerAttention(num_layers=8, feature_dim=128)
    x = torch.randn(4, 128, 200, 200)
    bev_features = [torch.randn(4, 128, 200, 200) for _ in range(8)]

    out = model(x, bev_features)

    print(f"✓ Input: {x.shape}")
    print(f"✓ BEV features: {len(bev_features)} layers")
    print(f"✓ Output: {out.shape}")

    assert out.shape == x.shape

    return model


def test_combined_pipeline():
    print("\n=== 测试组合流程 ===")

    # 模拟完整流程
    multiscale = MultiScaleConv(9, 128)
    spatial_attn = SpatialAttention()

    x = torch.randn(2, 9, 200, 200)

    # Step 1: 多尺度卷积
    features = multiscale(x)
    print(f"After MultiScaleConv: {features.shape}")

    # Step 2: 空间注意力
    features = spatial_attn(features)
    print(f"After SpatialAttention: {features.shape}")

    print("✓ Combined pipeline works!")


def main():
    print("=" * 60)
    print("Day 4: 模块测试")
    print("=" * 60)

    test_multiscale_conv()
    test_spatial_attention()
    test_cross_layer_attention()
    test_combined_pipeline()

    print("\n" + "=" * 60)
    print("✓ Day 4 所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
