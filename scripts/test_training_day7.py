"""
Day 7 测试脚本：验证训练流程
用法: python scripts/test_training_day7.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from models.retrieval_net import RetrievalNet
from training.losses import TripletLoss, InfoNCELoss
from training.trainer import BaseTrainer


class DummyDataset(Dataset):
    """模拟数据集（用于快速测试）"""

    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 模拟BEV数据 [9, 200, 200]
        anchor = torch.randn(9, 200, 200)
        positive = anchor + 0.1 * torch.randn(9, 200, 200)  # 相似
        negatives = torch.randn(10, 9, 200, 200)  # 10个负样本

        return {
            'anchor': anchor,
            'positive': positive,
            'negatives': negatives,
            'anchor_idx': idx,
            'positive_idx': idx,
            'negative_indices': list(range(10))
        }


def test_losses():
    """测试损失函数"""
    print("=" * 60)
    print("测试 1/4: 损失函数")
    print("=" * 60)

    batch_size = 4
    feature_dim = 128
    num_negatives = 10

    anchor = torch.randn(batch_size, feature_dim)
    positive = anchor + 0.1 * torch.randn(batch_size, feature_dim)
    negatives = torch.randn(batch_size, num_negatives, feature_dim)

    # 测试Triplet Loss
    print("\n[1] Triplet Loss (Hard Mining)")
    triplet_loss = TripletLoss(margin=0.5, mining='hard')
    loss, stats = triplet_loss(anchor, positive, negatives)

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Pos dist mean: {stats['pos_dist_mean']:.4f}")
    print(f"  Neg dist mean: {stats['neg_dist_mean']:.4f}")
    print(f"  Active triplets: {stats['active_triplets']:.2%}")

    # 测试InfoNCE Loss
    print("\n[2] InfoNCE Loss")
    infonce_loss = InfoNCELoss(temperature=0.07)
    loss, stats = infonce_loss(anchor, positive, negatives)

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Accuracy: {stats['accuracy']:.2%}")
    print(f"  Pos sim mean: {stats['pos_sim_mean']:.4f}")
    print(f"  Neg sim mean: {stats['neg_sim_mean']:.4f}")

    print("\n✓ 损失函数测试通过")


def test_trainer():
    """测试训练器"""
    print("\n" + "=" * 60)
    print("测试 2/4: 训练器基类")
    print("=" * 60)

    # 创建模型
    model = RetrievalNet(output_dim=128)

    # 创建模拟数据
    train_dataset = DummyDataset(size=16)
    val_dataset = DummyDataset(size=8)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # 创建优化器和损失
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = TripletLoss(margin=0.5, mining='hard')

    # 创建训练器
    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        criterion=criterion,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        log_dir='logs',
        checkpoint_dir='checkpoints',
        experiment_name='test_day7'
    )

    print("\n[1] 测试单个epoch训练")
    train_metrics = trainer.train_epoch()
    print(f"  Train loss: {train_metrics['loss']:.4f}")

    print("\n[2] 测试验证")
    val_metrics = trainer.validate()
    print(f"  Val loss: {val_metrics['loss']:.4f}")

    print("\n[3] 测试checkpoint保存")
    trainer.save_checkpoint(val_metrics['loss'], is_best=True)
    print(f"  ✓ Checkpoint saved")

    print("\n✓ 训练器测试通过")


def test_full_training_loop():
    """测试完整训练循环（1个epoch）"""
    print("\n" + "=" * 60)
    print("测试 3/4: 完整训练循环（1 epoch）")
    print("=" * 60)

    # 创建模型
    model = RetrievalNet(output_dim=128)

    # 创建数据
    train_dataset = DummyDataset(size=16)
    val_dataset = DummyDataset(size=8)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # 创建优化器、调度器、损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    criterion = TripletLoss(margin=0.5, mining='hard')

    # 创建训练器
    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        log_dir='logs',
        checkpoint_dir='checkpoints',
        experiment_name='test_full_loop'
    )

    # 训练1个epoch
    print("\n开始训练...")
    trainer.train(num_epochs=1)

    print("\n✓ 完整训练循环测试通过")


def test_with_real_data():
    """使用真实数据测试（如果可用）"""
    print("\n" + "=" * 60)
    print("测试 4/4: 真实数据测试（可选）")
    print("=" * 60)

    # 检查是否有真实数据
    cache_dir = "/home/wzj/pan1/contour_deep/data/Chilean_BEV_Cache/"
    train_pickle = "/home/wzj/pan1/contour_deep/data/test_queries_chilean_period.pickle"

    if not os.path.exists(cache_dir) or not os.path.exists(train_pickle):
        print("  ⚠ 真实数据不可用，跳过此测试")
        return

    from data.dataset_retrieval import RetrievalDataset, create_dataloader

    # 加载配置
    config_path = "/home/wzj/pan1/contour_deep/configs/config_base.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 创建数据集（只用少量数据测试）
    train_dataset = RetrievalDataset(
        queries_pickle=train_pickle,
        cache_root=cache_dir,
        split='train',
        num_negatives=10,
        augmentation_config=config['augmentation'],
        resolution=config['bev']['resolution'],
        use_cache=True,
        max_cache_size=100
    )

    # 只取前16个样本测试
    train_dataset.query_keys = train_dataset.query_keys[:16]

    train_loader = create_dataloader(
        train_dataset,
        batch_size=4,
        num_workers=2,
        shuffle=True
    )

    # 创建模型
    model = RetrievalNet(output_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = TripletLoss(margin=0.5, mining='hard')

    # 创建训练器
    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=train_loader,  # 用训练集作为验证集（仅测试）
        optimizer=optimizer,
        scheduler=None,
        criterion=criterion,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        log_dir='logs',
        checkpoint_dir='checkpoints',
        experiment_name='test_real_data'
    )

    print("\n使用真实Chilean数据训练1个epoch...")
    trainer.train(num_epochs=1)

    print("\n✓ 真实数据测试通过")


def main():
    print("\n" + "=" * 60)
    print("Day 7: 训练框架测试")
    print("=" * 60)

    # 测试1: 损失函数
    test_losses()

    # 测试2: 训练器基类
    test_trainer()

    # 测试3: 完整训练循环
    test_full_training_loop()

    # 测试4: 真实数据（可选）
    test_with_real_data()

    print("\n" + "=" * 60)
    print("✓ Day 7 所有测试完成！")
    print("=" * 60)

    print("\n下一步:")
    print("  1. 运行完整训练: python training/train_retrieval.py")
    print("  2. 监控训练: tensorboard --logdir logs")
    print("  3. Day 8-10: 首次完整训练（50 epochs）")


if __name__ == "__main__":
    main()
