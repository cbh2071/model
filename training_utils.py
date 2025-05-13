# -*- coding: utf-8 -*-
"""
训练和评估相关函数。
"""
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score # accuracy_score 在多标签中是 exact match ratio
from tqdm.auto import tqdm
from sklearn.preprocessing import MultiLabelBinarizer


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    device: torch.device,
    model_name: str = "Model",
    patience: int = 5, # 新增：Early Stopping 的耐心值
    min_delta: float = 0.001, # 新增：认为损失改善的最小变化量
    checkpoint_dir: str = "checkpoints", # 新增：检查点保存目录
    start_epoch: int = 0 # 新增：从哪个epoch开始 (用于继续训练)
) -> Tuple[List[float], List[float], nn.Module]: # 返回训练好的最佳模型
    """训练单个模型，并实现 Early Stopping。"""
    print(f"\n--- 开始训练 {model_name} (Early Stopping: patience={patience}, min_delta={min_delta}) ---")
    model.to(device)
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state_dict = model.state_dict() # 初始化最佳模型状态
    os.makedirs(checkpoint_dir, exist_ok=True) # 创建检查点目录

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [训练]", leave=False)
        for features, labels in train_pbar:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)
            train_pbar.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # --- 验证阶段 ---
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
        
        # --- 保存当前 epoch 的检查点 ---
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'best_val_loss': best_val_loss, # 可以保存当前的最佳验证损失
            # 'best_model_state_dict': best_model_state_dict # 如果也想保存最佳模型在检查点中
        }, checkpoint_path)
        print(f"  (检查点已保存到 {checkpoint_path})")

        # --- Early Stopping 检查 ---
        if epoch_val_loss < best_val_loss - min_delta: # 损失是否有显著改善
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_model_state_dict = model.state_dict() # 保存当前最佳模型的状态
            print(f"  (验证损失改善至 {best_val_loss:.4f}，保存模型状态)")
        else:
            epochs_no_improve += 1
            print(f"  (验证损失未显著改善，连续 {epochs_no_improve} epoch)")

        if epochs_no_improve >= patience:
            print(f"Early stopping触发：连续 {patience} 个 epochs 验证损失未改善。")
            break # 提前结束训练

    print(f"--- {model_name} 训练完成 ---")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    # 加载最佳模型权重
    model.load_state_dict(best_model_state_dict)
    return train_losses, val_losses, model # 返回加载了最佳权重的模型


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