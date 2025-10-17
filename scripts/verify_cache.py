"""
验证BEV缓存完整性
用法: python scripts/verify_cache.py
"""

import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

cache_root = "/home/wzj/pan1/contour_deep/data/Chilean_BEV_Cache/"

for split in ['train', 'test']:
    cache_dir = os.path.join(cache_root, split)
    cache_files = list(Path(cache_dir).glob("*.npz"))

    print(f"\n验证 {split} 集: {len(cache_files)} 个文件")

    corrupted = []
    for cache_file in tqdm(cache_files, desc=f"Checking {split}"):
        try:
            data = np.load(cache_file)
            assert data['bev_layers'].shape == (8, 200, 200)
            assert data['vcd'].shape == (200, 200)
            assert data['bev_layers'].dtype == np.uint8
            assert data['vcd'].dtype == np.uint8
        except Exception as e:
            corrupted.append((cache_file.name, str(e)))

    if corrupted:
        print(f"  ✗ 发现 {len(corrupted)} 个损坏文件:")
        for name, error in corrupted[:10]:
            print(f"    - {name}: {error}")
    else:
        print(f"  ✓ 所有文件验证通过!")
