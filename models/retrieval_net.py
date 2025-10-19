"""
RetrievalNet: 方向1检索特征提取网络
输入: [B, 9, 200, 200] (8层BEV + 1层VCD)
输出: [B, 128] 全局特征向量
"""

import torch
import torch.nn as nn
from models.modules import MultiScaleConv, SpatialAttention, CrossLayerAttention


class ResBlock(nn.Module):
    """残差块，用于下采样"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 捷径连接（stride>1时需要下采样）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接
        out += self.shortcut(x)
        out = self.relu(out)

        return out


class RetrievalNet(nn.Module):
    """
    方向1: 检索特征提取网络

    流程:
    Input [B,9,200,200]
    → MultiScaleConv [B,128,200,200]
    → SpatialAttention [B,128,200,200]
    → ResBlock×3 (下采样到25×25)
    → CrossLayerAttention (融合8层BEV)
    → GlobalPooling [B,256]
    → FC [B,128]
    """

    def __init__(self, output_dim=128):
        super(RetrievalNet, self).__init__()

        # 阶段1: 多尺度卷积 (9→128通道，尺寸不变)
        self.multiscale_conv = MultiScaleConv(in_channels=9, out_channels=128)

        # 阶段2: 空间注意力
        self.spatial_attention = SpatialAttention()

        # 阶段3: 残差下采样块
        # 200×200 → 100×100
        self.res_block1 = ResBlock(128, 128, stride=2)
        # 100×100 → 50×50
        self.res_block2 = ResBlock(128, 128, stride=2)
        # 50×50 → 25×25
        self.res_block3 = ResBlock(128, 128, stride=2)

        # 阶段4: 跨层注意力（融合8层BEV信息）
        self.cross_layer_attention = CrossLayerAttention(num_layers=8, feature_dim=128)

        # 阶段5: 全局池化
        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.gmp = nn.AdaptiveMaxPool2d(1)  # 全局最大池化

        # 阶段6: 全连接层
        # 拼接GAP和GMP: 128+128=256
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: [B, 9, 200, 200] 输入BEV (8层+VCD)

        Returns:
            features: [B, 128] 全局特征向量
        """
        batch_size = x.size(0)

        # 阶段1: 多尺度卷积
        x = self.multiscale_conv(x)  # [B, 128, 200, 200]

        # 阶段2: 空间注意力
        x = self.spatial_attention(x)  # [B, 128, 200, 200]

        # 阶段3: 残差下采样
        # 保存每层特征用于跨层注意力
        layer_features = []

        # 分离8层BEV特征（假设前8个通道对应8层，实际应根据网络设计调整）
        # 注意: 这里简化处理，实际应在MultiScaleConv后就分离
        # 为演示，我们直接使用当前特征的8个切片
        for i in range(8):
            # 提取每16个通道作为一层的特征（128/8=16）
            start_ch = i * 16
            end_ch = (i + 1) * 16
            layer_feat = x[:, start_ch:end_ch, :, :]
            layer_features.append(layer_feat)

        # 下采样
        x = self.res_block1(x)  # [B, 128, 100, 100]
        x = self.res_block2(x)  # [B, 128, 50, 50]
        x = self.res_block3(x)  # [B, 128, 25, 25]

        # 阶段4: 跨层注意力融合
        # 需要将layer_features也下采样到25×25
        downsampled_layer_features = []
        downsample = nn.AdaptiveAvgPool2d(25)
        for lf in layer_features:
            downsampled_layer_features.append(downsample(lf))

        # 拼接回128通道
        layer_features_concat = torch.cat(downsampled_layer_features, dim=1)  # [B, 128, 25, 25]

        # 应用跨层注意力（使用分离的8层特征）
        x = self.cross_layer_attention(x, downsampled_layer_features)  # [B, 128, 25, 25]

        # 阶段5: 全局池化
        gap_features = self.gap(x)  # [B, 128, 1, 1]
        gmp_features = self.gmp(x)  # [B, 128, 1, 1]

        gap_features = gap_features.view(batch_size, -1)  # [B, 128]
        gmp_features = gmp_features.view(batch_size, -1)  # [B, 128]

        # 拼接
        global_features = torch.cat([gap_features, gmp_features], dim=1)  # [B, 256]

        # 阶段6: 全连接
        output = self.fc(global_features)  # [B, 128]

        # L2归一化（用于特征匹配）
        output = torch.nn.functional.normalize(output, p=2, dim=1)

        return output

    def get_embedding_dim(self):
        """返回输出特征维度"""
        return 128


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("Testing RetrievalNet...")

    # 创建网络
    model = RetrievalNet(output_dim=128)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n网络参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  参数量: {total_params / 1e6:.2f}M")

    # 测试前向传播
    print("\n测试前向传播:")
    batch_size = 4
    x = torch.randn(batch_size, 9, 200, 200)

    print(f"  输入: {x.shape}")

    # 前向传播
    with torch.no_grad():
        output = model(x)

    print(f"  输出: {output.shape}")
    print(f"  预期: [{batch_size}, 128]")

    assert output.shape == (batch_size, 128), "输出维度错误!"

    # 验证L2归一化
    norms = torch.norm(output, p=2, dim=1)
    print(f"\n  特征向量L2范数: {norms}")
    print(f"  范数均值: {norms.mean().item():.6f} (应接近1.0)")

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "L2归一化失败!"

    # 测试梯度
    print("\n测试梯度反向传播:")
    x.requires_grad = True
    output = model(x)
    loss = output.sum()
    loss.backward()

    print(f"  梯度检查: {'✓ 通过' if x.grad is not None else '✗ 失败'}")

    # 推理速度测试
    print("\n推理速度测试:")
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

    print(f"  平均推理时间: {avg_time:.2f}ms")
    print(f"  吞吐量: {fps:.1f} samples/sec")
    print(f"  目标: >30 FPS ({'✓ 达标' if fps > 30 else '✗ 未达标'})")

    print("\n✓ RetrievalNet所有测试通过!")
