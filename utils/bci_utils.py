"""
BCI Utilities: BCI生成和处理工具
功能：
1. 从轮廓生成BCI
2. BCI转换为图神经网络输入
"""

import numpy as np
from typing import List, Tuple, Dict
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.contour_types import (
    BCI, RelativePoint, NUM_BIN_KEY_LAYER, DIST_BIN_LAYERS
)

from utils.contour_view import ContourView


class BCIGenerator:
    """BCI生成器"""

    def __init__(self,
                 dist_firsts: int = 12,
                 neighbor_layer_range: int = 3,
                 bits_per_layer: int = 20,
                 min_dist: float = 1.0,
                 max_dist: float = 20.0,
                 resolution: float = 0.1):
        """
        初始化BCI生成器

        Args:
            dist_firsts: 每层搜索前N个最大轮廓作为潜在邻居
            neighbor_layer_range: 邻域层级范围（±N层）
            bits_per_layer: 每层的距离bins数量
            min_dist: 最小距离（米）
            max_dist: 最大距离（米）
        """
        self.dist_firsts = dist_firsts
        self.neighbor_layer_range = neighbor_layer_range
        self.bits_per_layer = bits_per_layer
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.resolution = resolution

    def generate_bci_for_contour(self,
                                  contours_all_layers: List[List[ContourView]],
                                  target_level: int,
                                  target_seq: int) -> BCI:
        """
        为指定轮廓生成BCI

        Args:
            contours_all_layers: 所有层的轮廓列表
            target_level: 目标轮廓的层级
            target_seq: 目标轮廓的序号

        Returns:
            bci: 生成的BCI
        """
        # 创建BCI
        bci = BCI(target_seq, target_level)

        # 获取中心轮廓位置
        if target_seq >= len(contours_all_layers[target_level]):
            return bci  # 返回空BCI

        v_cen = contours_all_layers[target_level][target_seq].pos_mean

        # 计算允许搜索的层级范围
        num_layers = len(contours_all_layers)
        min_layer = max(0, target_level - self.neighbor_layer_range)
        max_layer = min(num_layers - 1, target_level + self.neighbor_layer_range)

        # 遍历DIST_BIN_LAYERS中的层级
        for bl in range(NUM_BIN_KEY_LAYER):
            bit_offset = bl * self.bits_per_layer
            layer_idx = DIST_BIN_LAYERS[bl]

            # 检查该层级是否在允许的搜索范围内
            if layer_idx < min_layer or layer_idx > max_layer:
                continue

            # 边界检查
            if layer_idx >= len(contours_all_layers):
                continue

            # 搜索该层级的邻居
            search_count = min(self.dist_firsts, len(contours_all_layers[layer_idx]))

            for j in range(search_count):
                # 排除自身
                if layer_idx != target_level or j != target_seq:
                    nei_contour = contours_all_layers[layer_idx][j]
                    vec_cc = nei_contour.pos_mean - v_cen # 像素向量：中心→邻居
                    pixel_dist = np.linalg.norm(vec_cc) # 像素距离
                    tmp_dist = pixel_dist * self.resolution  # 使用配置的分辨率，转换为米距离

                    # 距离范围检查
                    if tmp_dist <= self.min_dist or tmp_dist > self.max_dist - 1e-3:
                        continue

                    # 计算角度
                    tmp_orie = np.arctan2(vec_cc[1], vec_cc[0])

                    # 计算距离bin索引
                    bin_width = (self.max_dist - self.min_dist) / self.bits_per_layer
                    dist_idx = min(
                        int(np.floor((tmp_dist - self.min_dist) / bin_width)),
                        self.bits_per_layer - 1
                    )
                    dist_idx += bit_offset


                    # 更新BCI
                    if dist_idx < self.bits_per_layer * NUM_BIN_KEY_LAYER:
                        bci.dist_bin[dist_idx] = True
                        #dist_bin就是一维bool向量，[False, True, ...];
                        #False或True的位置是按距离由近到远排序的，也就是距离bin索引dist_idx
                        bci.nei_pts.append(
                            RelativePoint(layer_idx, j, dist_idx, tmp_dist, tmp_orie)
                        )#nei_pts是某一个轮廓（作为中心轮廓）全部的bci邻居信息：所属层、所属层轮廓序号、与中心轮廓的距离bin索引、与中心轮廓的实际距离、与中心轮廓的实际角度

        # 排序并建立索引段
        if bci.nei_pts:
            bci.nei_pts.sort(key=lambda p: p.bit_pos) #bit_pos就是dist_idx
            bci.nei_idx_segs = [0]
            for p1 in range(len(bci.nei_pts)):
                if bci.nei_pts[bci.nei_idx_segs[-1]].bit_pos != bci.nei_pts[p1].bit_pos:
                    bci.nei_idx_segs.append(p1)
            bci.nei_idx_segs.append(len(bci.nei_pts)) #为了查找同一距离bin的所有邻居，

        return bci

    def generate_all_bcis(self,
                          contours_all_layers: List[List[ContourView]],
                          piv_firsts: int = 12) -> List[List[BCI]]:
        """
        为所有轮廓生成BCI

        Args:
            contours_all_layers: 所有层的轮廓列表
            piv_firsts: 每层生成前N个轮廓的BCI

        Returns:
            all_bcis: 每层的BCI列表
        """
        num_layers = len(contours_all_layers)
        all_bcis = []

        for level in range(num_layers):
            layer_bcis = []
            generate_count = min(piv_firsts, len(contours_all_layers[level]))

            for seq in range(generate_count):
                bci = self.generate_bci_for_contour(contours_all_layers, level, seq)
                layer_bcis.append(bci)

            all_bcis.append(layer_bcis)
            print(f"  Layer {level}: generated {len(layer_bcis)} BCIs")

        return all_bcis


def bci_to_graph_data(bci: BCI) -> Dict:
    """
    将BCI转换为图神经网络输入格式

    Args:
        bci: BCI对象

    Returns:
        graph_data: 包含节点特征和邻接信息的字典
            - node_features: [N, F] 节点特征矩阵
            - edge_index: [2, E] 边索引
            - edge_weight: [E] 边权重
            - num_nodes: 节点数量
    """
    num_neighbors = len(bci.nei_pts)

    if num_neighbors == 0:
        # 空BCI（没有邻居）
        return {
            'node_features': np.zeros((1, 27), dtype=np.float32),  # 只有中心节点
            'edge_index': np.array([[0], [0]], dtype=np.int64),  # 自环
            'edge_weight': np.array([1.0], dtype=np.float32),
            'num_nodes': 1
        }

    # 节点数量 = 中心节点 + 邻居节点
    num_nodes = 1 + num_neighbors

    # 构建节点特征 node_features[num_nodes, 27]；num_nodes=0是中心节点特征，num_node非0是邻居节点特征；
    # 特征维度: 层级one-hot(8) + 距离(1) + 角度sin/cos(2) + 占位(16) = 27
    node_features = np.zeros((num_nodes, 27), dtype=np.float32)

    # 中心节点特征（索引0）
    if bci.level < 8:
        node_features[0, bci.level] = 1.0  # one-hot编码
    # 中心节点距离和角度都是0

    # 邻居节点特征（索引1到num_neighbors）
    for i, nei_pt in enumerate(bci.nei_pts):
        node_idx = i + 1

        # 层级one-hot编码
        if nei_pt.level < 8:
            node_features[node_idx, nei_pt.level] = 1.0

        # 归一化距离 (1-20米归一化到0-1)
        normalized_dist = (nei_pt.r - 1.0) / 19.0
        node_features[node_idx, 8] = normalized_dist

        # 角度的sin和cos
        node_features[node_idx, 9] = np.sin(nei_pt.theta)
        node_features[node_idx, 10] = np.cos(nei_pt.theta)

        # 占位特征（维度11-26，暂时为0，将来可以添加轮廓特征）

    # 构建边索引（全连接图）：每个节点都连接到其他所有节点（全连接图）。
    edge_list = []
    edge_weights = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # 排除自环
                edge_list.append([i, j])

                # 边权重：中心到邻居的权重基于距离
                if i == 0:  # 中心节点到邻居
                    dist = bci.nei_pts[j-1].r if j > 0 else 1.0
                    weight = 1.0 / (dist + 1.0)
                elif j == 0:  # 邻居到中心节点
                    dist = bci.nei_pts[i-1].r
                    weight = 1.0 / (dist + 1.0)
                else:  # 邻居之间
                    # 计算邻居之间的角度差异
                    theta1 = bci.nei_pts[i-1].theta
                    theta2 = bci.nei_pts[j-1].theta
                    angle_diff = abs(theta1 - theta2)
                    if angle_diff > np.pi:
                        angle_diff = 2 * np.pi - angle_diff
                    weight = 1.0 - angle_diff / np.pi  # 角度越接近权重越大

                edge_weights.append(weight)

    edge_index = np.array(edge_list, dtype=np.int64).T  # [2, E]
    edge_weight = np.array(edge_weights, dtype=np.float32)

    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'num_nodes': num_nodes,
        'center_level': bci.level,
        'center_seq': bci.piv_seq
    }


def get_bci_statistics(all_bcis: List[List[BCI]]) -> Dict:
    """
    计算BCI统计信息

    Args:
        all_bcis: 每层的BCI列表

    Returns:
        stats: 统计字典
    """
    all_neighbor_counts = []
    all_distances = []
    cross_layer_count = 0
    total_neighbors = 0

    for layer_bcis in all_bcis:
        for bci in layer_bcis:
            num_neighbors = len(bci.nei_pts)
            all_neighbor_counts.append(num_neighbors)

            for nei_pt in bci.nei_pts:
                all_distances.append(nei_pt.r)
                total_neighbors += 1

                if nei_pt.level != bci.level:
                    cross_layer_count += 1

    stats = {
        'total_bcis': sum(len(layer) for layer in all_bcis),
        'avg_neighbors': np.mean(all_neighbor_counts) if all_neighbor_counts else 0,
        'std_neighbors': np.std(all_neighbor_counts) if all_neighbor_counts else 0,
        'min_neighbors': np.min(all_neighbor_counts) if all_neighbor_counts else 0,
        'max_neighbors': np.max(all_neighbor_counts) if all_neighbor_counts else 0,
        'avg_distance': np.mean(all_distances) if all_distances else 0,
        'std_distance': np.std(all_distances) if all_distances else 0,
        'cross_layer_ratio': cross_layer_count / total_neighbors if total_neighbors > 0 else 0,
    }

    # 邻居数量分布
    neighbor_dist = {
        '0': sum(1 for n in all_neighbor_counts if n == 0),
        '1-3': sum(1 for n in all_neighbor_counts if 1 <= n <= 3),
        '4-6': sum(1 for n in all_neighbor_counts if 4 <= n <= 6),
        '7-10': sum(1 for n in all_neighbor_counts if 7 <= n <= 10),
        '10+': sum(1 for n in all_neighbor_counts if n > 10),
    }

    stats['neighbor_distribution'] = neighbor_dist

    return stats


# ========== 测试代码 ==========
if __name__ == "__main__":
    """
    测试BCI生成器
    用法: python utils/bci_utils.py
    """
    print("Testing BCI Generator...")

    # 1. 加载真实的轮廓数据（从contour_utils.py生成的）
    print("\n准备测试数据...")

    # 先生成真实轮廓
    print("步骤1: 生成真实轮廓...")
    test_bev_path = "/home/wzj/pan1/contour_deep/data/test_bev_output.npz"
    if not os.path.exists(test_bev_path):
        print(f"Error: 找不到 {test_bev_path}")
        print("请先运行: python data/bev_generator.py")
        sys.exit(1)

    # 加载BEV数据
    loaded = np.load(test_bev_path)
    bev_layers = loaded['bev_layers']
    vcd = loaded['vcd']
    print(f"  加载BEV成功: {bev_layers.shape}")

    # 提取轮廓
    from utils.contour_utils import ContourExtractor
    from utils.contour_types import ContourViewStatConfig

    config = ContourViewStatConfig()
    extractor = ContourExtractor(config, min_contour_size=1)
    real_contours = extractor.extract_contours_from_all_layers(bev_layers)

    total_contours = sum(len(layer) for layer in real_contours)
    print(f"  提取轮廓成功: {total_contours}个轮廓，{len(real_contours)}层")

    # 显示每层轮廓数量
    for level, layer_contours in enumerate(real_contours):
        print(f"    Layer {level}: {len(layer_contours)} contours")

    # 2. 创建BCI生成器
    print("\n步骤2: 创建BCI生成器...")
    bci_generator = BCIGenerator(
        dist_firsts=12,
        neighbor_layer_range=3,
        bits_per_layer=20
    )
    print(f"  配置: dist_firsts=12, neighbor_range=±3, bits=20")

    # 3. 生成单个BCI测试（选择Layer 2的第一个轮廓，因为Layer 2轮廓最多）
    print("\n步骤3: 测试单个BCI生成...")

    # 找一个有较多轮廓的层级
    best_level = max(range(len(real_contours)), key=lambda i: len(real_contours[i]))
    print(f"  选择Layer {best_level}进行测试（该层有{len(real_contours[best_level])}个轮廓）")

    if len(real_contours[best_level]) > 0:
        test_seq = 0  # 选择第一个轮廓
        test_bci = bci_generator.generate_bci_for_contour(real_contours, target_level=best_level, target_seq=test_seq)

        print(f"\n  BCI信息:")
        print(f"    中心轮廓: Level {test_bci.level}, Seq {test_bci.piv_seq}")
        print(f"    邻居数量: {len(test_bci.nei_pts)}")
        print(f"    距离位激活数: {np.sum(test_bci.dist_bin)}")

        # 显示邻居详细信息（前5个）
        if len(test_bci.nei_pts) > 0:
            print(f"    前5个邻居详情:")
            for i, nei_pt in enumerate(test_bci.nei_pts[:5]):
                print(f"      邻居{i}: Level={nei_pt.level}, Seq={nei_pt.seq}, bit_pos={nei_pt.bit_pos}"
                      f"      距离={nei_pt.r:.2f}m, 角度={np.degrees(nei_pt.theta):.1f}°")

        # 4. 转换为图数据
        print("\n步骤4: 测试BCI到图数据转换...")
        graph_data = bci_to_graph_data(test_bci)
        print(f"  图数据信息:")
        print(f"    节点数量: {graph_data['num_nodes']}")
        print(f"    节点特征维度: {graph_data['node_features'].shape}")
        print(f"    边数量: {graph_data['edge_index'].shape[1]}")
        print(f"    边权重范围: [{graph_data['edge_weight'].min():.3f}, {graph_data['edge_weight'].max():.3f}]")

        # 显示节点特征统计
        node_features = graph_data['node_features']
        print(f"    节点特征统计:")
        print(f"      层级分布: {node_features[:, :8].sum(axis=0)}")  # one-hot层级
        print(f"      距离范围: [{node_features[:, 8].min():.3f}, {node_features[:, 8].max():.3f}]")
    else:
        print(f"  Warning: Layer {best_level} 没有轮廓，跳过测试")

    # 5. 生成所有BCI
    print("\n步骤5: 批量生成所有BCI...")
    all_bcis = bci_generator.generate_all_bcis(real_contours, piv_firsts=12)

    # 6. 统计信息
    print("\n步骤6: BCI统计分析...")
    stats = get_bci_statistics(all_bcis)
    print(f"  总BCI数: {stats['total_bcis']}")
    print(f"  平均邻居数: {stats['avg_neighbors']:.1f} ± {stats['std_neighbors']:.1f}")
    print(f"  邻居数范围: [{stats['min_neighbors']}, {stats['max_neighbors']}]")
    print(f"  平均距离: {stats['avg_distance']:.2f} ± {stats['std_distance']:.2f} 米")
    print(f"  跨层连接比例: {stats['cross_layer_ratio']:.1%}")
    print(f"  邻居分布:")
    for name, count in stats['neighbor_distribution'].items():
        ratio = count / stats['total_bcis'] * 100 if stats['total_bcis'] > 0 else 0
        print(f"    {name}邻居: {count} ({ratio:.1f}%)")

    # 7. 每层BCI详情
    print("\n步骤7: 每层BCI详情...")
    for level, layer_bcis in enumerate(all_bcis):
        if len(layer_bcis) > 0:
            neighbor_counts = [len(bci.nei_pts) for bci in layer_bcis]
            avg_neighbors = np.mean(neighbor_counts)

            # 统计跨层连接
            cross_layer = 0
            total_neighbors = 0
            for bci in layer_bcis:
                for nei_pt in bci.nei_pts:
                    total_neighbors += 1
                    if nei_pt.level != level:
                        cross_layer += 1

            cross_ratio = cross_layer / total_neighbors * 100 if total_neighbors > 0 else 0

            print(f"  Layer {level}: {len(layer_bcis)} BCIs, "
                  f"平均邻居={avg_neighbors:.1f}, "
                  f"邻居范围=[{min(neighbor_counts)}, {max(neighbor_counts)}], "
                  f"跨层连接={cross_ratio:.1f}%")

    # 8. 保存测试结果（可选）
    print("\n步骤8: 保存测试结果...")
    test_output = {
        'contours': real_contours,
        'bcis': all_bcis,
        'stats': stats
    }

    import pickle

    with open('test_bci_output.pkl', 'wb') as f:
        pickle.dump(test_output, f)
    print(f"  测试结果已保存到: test_bci_output.pkl")

    print("\n" + "=" * 60)
    print("✓ BCI Generator测试完成!")
    print("=" * 60)

    # 9. 关键发现总结
    print("\n【关键发现】")
    if stats['avg_neighbors'] < 3:
        print(f"  ⚠️  警告: 平均邻居数偏低({stats['avg_neighbors']:.1f})，可能需要调整neighbor_layer_range")
    elif stats['avg_neighbors'] > 15:
        print(f"  ⚠️  警告: 平均邻居数偏高({stats['avg_neighbors']:.1f})，可能引入噪声")
    else:
        print(f"  ✓ 邻居数合理({stats['avg_neighbors']:.1f})，在Chilean场景最优范围(6-12)内")

    if stats['cross_layer_ratio'] > 0.7:
        print(f"  ✓ 跨层连接丰富({stats['cross_layer_ratio']:.1%})，符合Chilean巷道垂直结构特性")
    else:
        print(f"  ⚠️  跨层连接偏低({stats['cross_layer_ratio']:.1%})，可能需要增加neighbor_layer_range")

    zero_neighbor_ratio = stats['neighbor_distribution']['0'] / stats['total_bcis'] * 100 if stats[
                                                                                                 'total_bcis'] > 0 else 0
    if zero_neighbor_ratio > 50:
        print(f"  ⚠️  警告: {zero_neighbor_ratio:.1f}%的BCI没有邻居，检查dist_firsts和距离范围设置")
    else:
        print(f"  ✓ 零邻居BCI比例正常({zero_neighbor_ratio:.1f}%)")