"""
Contour Utilities: 轮廓提取和处理工具
功能：
1. 从BEV图像中提取轮廓
2. 轮廓统计和分析
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.contour_types import (
    ContourViewStatConfig,
    RunningStatRecorder
)
from utils.contour_view import ContourView


class ContourExtractor:
    """轮廓提取器"""

    def __init__(self, config: ContourViewStatConfig, min_contour_size: int = 1):
        """
        初始化轮廓提取器

        Args:
            config: 轮廓视图统计配置
            min_contour_size: 最小轮廓尺寸（像素数）
        """
        self.view_stat_cfg = config
        self.min_contour_size = min_contour_size
        self.cont_views = []  # 存储所有轮廓
        self.bev_pixfs = []  # BEV像素浮点信息

    def extract_contours_from_bev_layer(self,
                                        bev_layer: np.ndarray,
                                        level: int,
                                        bev_full: np.ndarray,
                                        bev_pixfs: List) -> List[ContourView]:
        """
        从单层BEV图像中提取轮廓

        Args:
            bev_layer: 单层BEV二值图 [H, W]，值为0或255
            level: 层级索引
            bev_full: 完整的BEV高度图 [H, W]
            bev_pixfs: BEV像素浮点信息列表

        Returns:
            contours: 轮廓列表
        """
        contours = []

        # 连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bev_layer, connectivity=8)
        # ** 作用 **：找出所有连通的白色像素区域。
        #
        # ** 输出 **：
        # - `num_labels`: 连通区域数量（包括背景，所以要减1）
        # - `labels`: 标签图
        # `[H, W]`，每个像素的值表示属于哪个连通组件
        # - 0 = 背景
        # - 1, 2, 3... = 各个轮廓
        #
        # - `stats`: 每个组件的统计
        # `[num_labels, 5]`
        # - `[n, cv2.CC_STAT_LEFT]`: 边界框左上角X
        # - `[n, cv2.CC_STAT_TOP]`: 边界框左上角Y
        # - `[n, cv2.CC_STAT_WIDTH]`: 宽度
        # - `[n, cv2.CC_STAT_HEIGHT]`: 高度
        # - `[n, cv2.CC_STAT_AREA]`: 面积（像素数）

        # 处理每个连通组件（跳过背景labels=0）
        for n in range(1, num_labels):
            if stats[n, cv2.CC_STAT_AREA] < self.min_contour_size:
                continue

            # 获取组件的边界框
            comp_x = stats[n, cv2.CC_STAT_LEFT]
            comp_y = stats[n, cv2.CC_STAT_TOP]
            comp_w = stats[n, cv2.CC_STAT_WIDTH]
            comp_h = stats[n, cv2.CC_STAT_HEIGHT]

            # 创建组件掩码：从标签图中提取当前组件的局部掩码
            mask_n = (labels[comp_y:comp_y + comp_h, comp_x:comp_x + comp_w] == n).astype(np.uint8)

            # 初始化统计记录器
            rec = RunningStatRecorder()
            poi_r, poi_c = -1, -1

            # 遍历组件内的每个像素
            for i in range(comp_h):
                for j in range(comp_w):
                    if mask_n[i, j]:
                        global_r = i + comp_y
                        global_c = j + comp_x
                        poi_r, poi_c = global_r, global_c

                        # 查找连续坐标
                        height = bev_full[global_r, global_c] if bev_full is not None else 0.0
                        rec.running_stats(float(global_r), float(global_c), height)

            if poi_r >= 0:
                # 创建轮廓视图
                contour = ContourView(level, poi_r, poi_c)
                contour.calc_stat_vals(rec, self.view_stat_cfg)
                contours.append(contour)

        # 按面积排序（从大到小）
        contours.sort(key=lambda x: x.cell_cnt, reverse=True)

        return contours

    def extract_contours_from_all_layers(self,
                                         bev_layers: np.ndarray,
                                         bev_full: Optional[np.ndarray] = None) -> List[List[ContourView]]:
        """
        从所有BEV层中提取轮廓

        Args:
            bev_layers: 多层BEV [num_layers, H, W]
            bev_full: 完整BEV高度图 [H, W]（可选）

        Returns:
            all_contours: 每层的轮廓列表 [[layer0_contours], [layer1_contours], ...]
        """
        num_layers = bev_layers.shape[0]
        all_contours = []

        for level in range(num_layers):
            layer_contours = self.extract_contours_from_bev_layer(
                bev_layers[level],
                level,
                bev_full,
                self.bev_pixfs
            )
            all_contours.append(layer_contours)
            print(f"  Layer {level}: extracted {len(layer_contours)} contours")

        return all_contours


def get_contour_statistics(contours: List[List[ContourView]]) -> dict:
    """
    计算轮廓统计信息

    Args:
        contours: 每层的轮廓列表

    Returns:
        stats: 统计字典
    """
    total_contours = sum(len(layer) for layer in contours)

    all_sizes = []
    all_eccentricities = []

    for layer in contours:
        for contour in layer:
            all_sizes.append(contour.cell_cnt)
            all_eccentricities.append(contour.eccen)

    stats = {
        'total_contours': total_contours,
        'avg_size': np.mean(all_sizes) if all_sizes else 0,
        'std_size': np.std(all_sizes) if all_sizes else 0,
        'min_size': np.min(all_sizes) if all_sizes else 0,
        'max_size': np.max(all_sizes) if all_sizes else 0,
        'avg_eccentricity': np.mean(all_eccentricities) if all_eccentricities else 0,
        'std_eccentricity': np.std(all_eccentricities) if all_eccentricities else 0,
    }

    # 尺寸分布统计
    size_bins = {
        'tiny': sum(1 for s in all_sizes if 1 <= s <= 5),
        'small': sum(1 for s in all_sizes if 6 <= s <= 15),
        'medium_small': sum(1 for s in all_sizes if 16 <= s <= 50),
        'medium': sum(1 for s in all_sizes if 51 <= s <= 150),
        'large': sum(1 for s in all_sizes if 151 <= s <= 500),
        'super_large': sum(1 for s in all_sizes if s > 500),
    }

    stats['size_distribution'] = size_bins

    return stats


# ========== 测试代码 ==========
if __name__ == "__main__":
    """
    测试轮廓提取器
    用法: python utils/contour_utils.py
    """
    print("Testing Contour Extractor...")

    # 1. 加载测试BEV数据
    test_bev_path = "/home/wzj/pan1/contour_deep/data/test_bev_output.npz"
    if not os.path.exists(test_bev_path):
        print(f"Error: 找不到测试BEV文件 {test_bev_path}")
        print("请先运行 python data/bev_generator.py 生成测试数据")
        sys.exit(1)

    loaded = np.load(test_bev_path)
    bev_layers = loaded['bev_layers']
    vcd = loaded['vcd']
    print(f"加载BEV成功: {bev_layers.shape}")

    # 2. 创建轮廓提取器
    config = ContourViewStatConfig()
    extractor = ContourExtractor(config, min_contour_size=1)

    # 3. 提取轮廓
    print("\n提取轮廓...")
    all_contours = extractor.extract_contours_from_all_layers(bev_layers)

    # 4. 统计信息
    print("\n轮廓统计:")
    stats = get_contour_statistics(all_contours)
    print(f"  总轮廓数: {stats['total_contours']}")
    print(f"  平均尺寸: {stats['avg_size']:.1f} ± {stats['std_size']:.1f} pixels")
    print(f"  尺寸范围: [{stats['min_size']}, {stats['max_size']}]")
    print(f"  平均偏心率: {stats['avg_eccentricity']:.3f}")
    print(f"  尺寸分布:")
    for name, count in stats['size_distribution'].items():
        print(f"    {name}: {count}")

    # 5. 每层详细信息
    print("\n每层轮廓详情:")
    for level, layer_contours in enumerate(all_contours):
        if len(layer_contours) > 0:
            sizes = [c.cell_cnt for c in layer_contours]
            print(f"  Layer {level}: {len(layer_contours)} contours, "
                  f"size range [{min(sizes)}, {max(sizes)}]")

    print("\n✓ Contour Extractor测试完成!")