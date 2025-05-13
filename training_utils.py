# -*- coding: utf-8 -*-
"""
训练和评估相关函数。
"""
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score # accuracy_score 在多标签中是 exact match ratio
from tqdm.auto import tqdm
from sklearn.preprocessing import MultiLabelBinarizer


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module, # 预期是 BCEWithLogitsLoss
    num_epochs: int,
    device: torch.device,
    model_name: str = "Model"
) -> Tuple[List[float], List[float]]:
    """训练单个模型。"""
    # (代码与之前版本基本相同，确保使用 BCEWithLogitsLoss)
    print(f"\n--- 开始训练 {model_name} ---")
    model.to(device)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [训练]", leave=False)

        for features, labels in train_pbar:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features) # Logits
            loss = criterion(outputs, labels)
            loss.backward()
            # 可选：梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * features.size(0)
            train_pbar.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [验证]", leave=False)
        with torch.no_grad():
            for features, labels in val_pbar:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
                val_pbar.set_postfix(loss=loss.item())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
             best_val_loss = epoch_val_loss
             # 可在这里添加保存最佳模型的逻辑
             # torch.save(model.state_dict(), f"{model_name}_best_val.pth")

    print(f"--- {model_name} 训练完成 ---")
    return train_losses, val_losses


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module, # 预期是 BCEWithLogitsLoss
    device: torch.device,
    mlb: MultiLabelBinarizer,
    threshold: float = 0.5
) -> Tuple[float, float, float, float, Dict[str, float]]:
    """
    评估模型在测试集上的表现 (多标签)。

    返回:
        - test_loss: 平均测试损失。
        - sample_f1: Sample-based F1 score。
        - micro_f1: Micro-averaged F1 score。
        - weighted_f1: Weighted F1 score。
        - class_f1: 字典 {class_name: f1_score}。
    """
    # (采纳 Script C 的评估逻辑，并增加 micro/weighted F1)
    print("\n--- 开始评估模型 ---")
    model.eval()
    model.to(device)

    total_loss = 0.0
    all_labels_np = []
    all_preds_np = []

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="评估", leave=False)
        for features, labels in test_pbar:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features) # Logits
            loss = criterion(outputs, labels)
            total_loss += loss.item() * features.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()

            all_labels_np.append(labels.cpu().numpy())
            all_preds_np.append(preds.cpu().numpy())

    test_loss = total_loss / len(test_loader.dataset)
    all_labels_np = np.concatenate(all_labels_np, axis=0)
    all_preds_np = np.concatenate(all_preds_np, axis=0)

    # 计算 F1 分数
    sample_f1 = f1_score(all_labels_np, all_preds_np, average='samples', zero_division=0)
    micro_f1 = f1_score(all_labels_np, all_preds_np, average='micro', zero_division=0)
    weighted_f1 = f1_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)

    # 计算每个类别的 F1
    class_f1_scores = f1_score(all_labels_np, all_preds_np, average=None, zero_division=0)
    class_names = mlb.classes_
    class_f1_dict = {name: score for name, score in zip(class_names, class_f1_scores)}

    # 计算 Exact Match Ratio (等同于多标签的 accuracy)
    exact_match_ratio = accuracy_score(all_labels_np, all_preds_np)

    print(f"评估完成:")
    print(f"  - 平均测试损失: {test_loss:.4f}")
    print(f"  - Exact Match Ratio: {exact_match_ratio:.4f}")
    print(f"  - Sample-based F1: {sample_f1:.4f}")
    print(f"  - Micro F1: {micro_f1:.4f}")
    print(f"  - Weighted F1: {weighted_f1:.4f}")
    # print(f"  - 各类别 F1 分数: {class_f1_dict}") # 可取消注释

    return test_loss, sample_f1, micro_f1, weighted_f1, class_f1_dict