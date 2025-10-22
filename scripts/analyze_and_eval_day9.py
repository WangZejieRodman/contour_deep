"""
Day 9: 评估Recall性能
用法: 在PyCharm中运行
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm


def evaluate_recall_on_testset():
    """在测试集上评估Recall"""
    print("=" * 60)
    print("在测试集上评估Recall@N")
    print("=" * 60)

    from models.retrieval_net import RetrievalNet
    from data.dataset_retrieval import RetrievalDataset
    import yaml
    from sklearn.neighbors import NearestNeighbors

    # 1. 加载模型
    print("\n加载模型...")
    model = RetrievalNet(output_dim=128)
    checkpoint = torch.load('checkpoints/retrieval_baseline_day8/best.pth',
                            map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()
    print(f"  加载Epoch {checkpoint['epoch'] + 1}的模型")
    print(f"  Val Loss: {checkpoint['metric']:.4f}")

    # 2. 加载配置
    with open('/home/wzj/pan1/contour_deep/configs/config_base.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 3. 创建数据库（训练集）
    print("\n创建数据库特征（训练集）...")
    train_dataset = RetrievalDataset(
        queries_pickle='/home/wzj/pan1/contour_deep/data/training_queries_chilean_period.pickle',
        cache_root='/home/wzj/pan1/contour_deep/data/Chilean_BEV_Cache',
        split='train',
        num_negatives=10,
        augmentation_config=None,  # 评估时不增强
        resolution=config['bev']['resolution'],
        use_cache=True
    )

    database_features = []
    database_keys = []

    with torch.no_grad():
        for i, query_key in enumerate(tqdm(train_dataset.query_keys, desc="提取数据库特征")):
            bev_data = train_dataset._load_bev_from_cache(query_key)
            if bev_data is None:
                continue

            bev_tensor = train_dataset._preprocess_bev(*bev_data, apply_aug=False)
            bev_tensor = bev_tensor.unsqueeze(0).cuda()

            feat = model(bev_tensor)
            database_features.append(feat.cpu().numpy())
            database_keys.append(query_key)

    database_features = np.vstack(database_features)
    print(f"  数据库大小: {len(database_features)}")

    # 4. 创建测试集
    print("\n创建查询特征（测试集）...")
    test_dataset = RetrievalDataset(
        queries_pickle='/home/wzj/pan1/contour_deep/data/test_queries_chilean_period.pickle',
        cache_root='/home/wzj/pan1/contour_deep/data/Chilean_BEV_Cache',
        split='test',
        num_negatives=10,
        augmentation_config=None,
        resolution=config['bev']['resolution'],
        use_cache=True
    )

    query_features = []
    query_keys = []
    query_ground_truths = []

    with torch.no_grad():
        for i, query_key in enumerate(tqdm(test_dataset.query_keys, desc="提取查询特征")):
            bev_data = test_dataset._load_bev_from_cache(query_key)
            if bev_data is None:
                continue

            bev_tensor = test_dataset._preprocess_bev(*bev_data, apply_aug=False)
            bev_tensor = bev_tensor.unsqueeze(0).cuda()

            feat = model(bev_tensor)
            query_features.append(feat.cpu().numpy())
            query_keys.append(query_key)

            # 获取ground truth
            query_data = test_dataset.queries[query_key]
            query_ground_truths.append(set(query_data['positives']))

    query_features = np.vstack(query_features)
    print(f"  查询集大小: {len(query_features)}")

    # 5. 构建KNN索引
    print("\n构建KNN索引...")
    knn = NearestNeighbors(n_neighbors=min(25, len(database_features)),
                           metric='euclidean',
                           n_jobs=-1)
    knn.fit(database_features)

    # 6. 计算Recall@1, @5, @10, @25
    print("\n计算Recall@N...")
    recalls = {1: 0, 5: 0, 10: 0, 25: 0}
    valid_queries = 0

    for i in tqdm(range(len(query_features)), desc="KNN检索"):
        if len(query_ground_truths[i]) == 0:
            continue

        valid_queries += 1

        # KNN搜索
        distances, indices = knn.kneighbors([query_features[i]])
        retrieved_keys = [database_keys[idx] for idx in indices[0]]

        # 检查Recall@K
        for k in [1, 5, 10, 25]:
            if k > len(retrieved_keys):
                continue
            if any(key in query_ground_truths[i] for key in retrieved_keys[:k]):
                recalls[k] += 1

    # 归一化
    for k in recalls:
        recalls[k] = recalls[k] / valid_queries * 100

    # 7. 输出结果
    print("\n" + "=" * 60)
    print("评估结果:")
    print("=" * 60)
    print(f"有效查询数: {valid_queries}")
    print(f"数据库大小: {len(database_features)}")
    print(f"\nRecall性能:")
    print(f"  Recall@1:  {recalls[1]:.2f}%")
    print(f"  Recall@5:  {recalls[5]:.2f}%")
    print(f"  Recall@10: {recalls[10]:.2f}%")
    print(f"  Recall@25: {recalls[25]:.2f}%")

    # 8. 与baseline对比
    print("\n" + "=" * 60)
    print("与Baseline对比:")
    print("=" * 60)
    print(f"  Contour Context (baseline): 73.14%")
    print(f"  当前方法 (方向1):           {recalls[1]:.2f}%")
    print(f"  差距:                        {recalls[1] - 73.14:.2f}%")

    if recalls[1] > 50:
        print(f"\n  ✓ 达到初步目标 (>50%)")
        print(f"  建议: 继续优化以接近baseline")
    elif recalls[1] > 30:
        print(f"\n  ⚠️  性能一般 (30-50%)")
        print(f"  建议: 需要调整模型结构或超参数")
    else:
        print(f"\n  ❌ 性能较差 (<30%)")
        print(f"  建议: 检查数据、模型、损失函数")

    return recalls


def analyze_failure_cases():
    """分析失败案例（可选）"""
    print("\n" + "=" * 60)
    print("失败案例分析（待实现）")
    print("=" * 60)
    print("建议分析:")
    print("  1. 特征空间可视化（t-SNE）")
    print("  2. 距离分布统计")
    print("  3. 困难样本分析")


def main():
    """主函数"""
    print("=" * 60)
    print("Day 9: 训练结果评估")
    print("=" * 60)

    # 1. 检查checkpoint
    checkpoint_path = "checkpoints/retrieval_baseline_day8/best.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint不存在: {checkpoint_path}")
        return

    # 2. 评估Recall
    try:
        recalls = evaluate_recall_on_testset()

        # 3. 保存结果
        import json
        results = {
            'recalls': recalls,
            'checkpoint': checkpoint_path,
            'baseline': 73.14
        }

        os.makedirs('logs/day9_evaluation', exist_ok=True)
        with open('logs/day9_evaluation/day9_evaluation.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n结果已保存: logs/day9_evaluation/day9_evaluation.json")

    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()

    # 4. 下一步建议
    print("\n" + "=" * 60)
    print("下一步:")
    print("=" * 60)
    if 'recalls' in locals() and recalls[1] > 50:
        print("  1. Day 10: 尝试超参数优化")
        print("  2. 增加训练epochs（如20→30）")
        print("  3. 尝试不同的损失函数")
    else:
        print("  1. 检查模型结构（是否需要调整）")
        print("  2. 检查损失函数（Triplet Loss参数）")
        print("  3. 检查数据增强策略")


if __name__ == "__main__":
    main()
