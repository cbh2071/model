# -*- coding: utf-8 -*-
"""
定义 PyTorch Dataset 类。
"""
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

class ProteinFeatureDataset(Dataset):
    """存储预计算特征和标签的数据集"""
    def __init__(self, protein_ids: List[str], features: List[np.ndarray], labels: np.ndarray):
        """
        Args:
            protein_ids (List[str]): 蛋白质 ID 列表。
            features (List[np.ndarray]): 预计算的特征向量列表。
            labels (np.ndarray): 编码后的标签矩阵 (Numpy array, float32)。
        """
        assert len(protein_ids) == len(features) == len(labels), "ID, 特征和标签数量必须一致"
        self.protein_ids = protein_ids
        self.features = features # 保持为列表，在 getitem 中转换
        self.labels = labels

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回指定索引的特征和标签张量。"""
        # 在这里将 numpy array 转换为 tensor
        feature = torch.from_numpy(self.features[idx]).float()
        label = torch.from_numpy(self.labels[idx]).float()
        return feature, label