# -*- coding: utf-8 -*-
"""
主执行脚本：基于预处理数据进行模型训练。
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
from typing import List, Tuple

# --- 导入自定义模块 ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import config
from datasets import ProteinFeatureDataset
from models import BiLSTMAttention, CNN_BiLSTM, EnsembleModel
from training_utils import train_model

def load_prepared_data(data_dir: str, split_name: str) -> Tuple[List[str], List[np.ndarray], np.ndarray]:
    """加载预处理和划分好的数据"""
    ids_path = os.path.join(data_dir, f"{split_name}_ids.npy")
    features_path = os.path.join(data_dir, f"{split_name}_features.npy")
    labels_path = os.path.join(data_dir, f"{split_name}_labels.npy")

    if not all(os.path.exists(p) for p in [ids_path, features_path, labels_path]):
        raise FileNotFoundError(f"错误: {split_name} 的数据文件在 {data_dir} 中未完全找到。请先运行 prepare_data_and_split.py。")

    ids = np.load(ids_path, allow_pickle=True).tolist()
    features_loaded = np.load(features_path, allow_pickle=True)
    features_list = [feat for feat in features_loaded]
    labels = np.load(labels_path, allow_pickle=True)
    return ids, features_list, labels

def main(args):
    """主执行流程 (训练模型)"""
    print("--- 基于预处理数据进行模型训练 ---")
    print(f"使用设备: {config.DEVICE}")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # --- 步骤 1: 加载预处理和划分好的训练集和验证集数据 ---
    print(f"\n--- 从 '{args.prepared_data_dir}' 加载训练和验证数据 ---")
    try:
        train_ids, train_features, train_labels = load_prepared_data(args.prepared_data_dir, "train")
        val_ids, val_features, val_labels = load_prepared_data(args.prepared_data_dir, "validation")
    except FileNotFoundError as e:
        print(e)
        return

    if not train_ids or not val_ids:
        print("错误：未能加载训练或验证数据。")
        return

    # 加载 MLB 编码器
    mlb_path = os.path.join(args.prepared_data_dir, "mlb_encoder.pkl")
    if not os.path.exists(mlb_path):
        print(f"错误: MLB 编码器文件未找到于 {mlb_path}")
        return
    with open(mlb_path, 'rb') as f:
        mlb = pickle.load(f)
    num_classes = len(mlb.classes_)
    print(f"MLB 加载完成，类别数: {num_classes}, 类别: {mlb.classes_}")

    # --- 步骤 2: 创建 Dataset 和 DataLoader ---
    print("\n--- 创建 Dataset 和 DataLoader ---")
    train_dataset = ProteinFeatureDataset(train_ids, train_features, train_labels)
    val_dataset = ProteinFeatureDataset(val_ids, val_features, val_labels)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    print(f"数据集加载: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}")

    # --- 步骤 3: 计算 pos_weight ---
    pos_weight_tensor = None
    if train_labels.size > 0 and train_labels.shape[1] > 0:
        print("\n--- 基于加载的训练集计算 pos_weight ---")
        num_train_samples = train_labels.shape[0]
        num_positive_train = train_labels.sum(axis=0)
        num_negative_train = num_train_samples - num_positive_train
        epsilon = 1e-6
        pos_weight_values_train = num_negative_train / (num_positive_train + epsilon)
        pos_weight_values_train = np.clip(pos_weight_values_train, a_min=1.0, a_max=100.0)
        
        for i in range(len(pos_weight_values_train)):
            if num_positive_train[i] == 0:
                print(f"警告: 类别 '{mlb.classes_[i]}' 在训练集中没有正样本。pos_weight 将被设为 1.0。")
                pos_weight_values_train[i] = 1.0
            elif num_negative_train[i] == 0:
                print(f"警告: 类别 '{mlb.classes_[i]}' 在训练集中没有负样本。pos_weight 将被设为 1.0。")
                pos_weight_values_train[i] = 1.0

        alpha = config.ALPHA
        pos_weight_values_train_scaled = pos_weight_values_train * alpha
        pos_weight_tensor = torch.tensor(pos_weight_values_train_scaled, dtype=torch.float32).to(config.DEVICE)
        print(f"基于训练集计算得到的 pos_weight (scaled, 前10个): {pos_weight_values_train_scaled[:10]}")
    else:
        print("警告: 训练集标签无效，无法计算 pos_weight。")

    # --- 步骤 4: 模型初始化与训练 ---
    print("\n--- 模型初始化与训练 ---")
    if pos_weight_tensor is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        print("损失函数 BCEWithLogitsLoss 已配置类别权重 (pos_weight)。")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("损失函数 BCEWithLogitsLoss 未配置类别权重。")

    # --- 训练 BiLSTM+Attention ---
    model_bilstm = BiLSTMAttention(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=num_classes,
        num_layers=config.NUM_LSTM_LAYERS
    )
    model_bilstm.to(config.DEVICE)
    optimizer_bilstm = torch.optim.Adam(model_bilstm.parameters(), lr=args.learning_rate)

    # 添加学习率调度器
    scheduler_bilstm = None
    if args.lr_scheduler == 'step':
        scheduler_bilstm = torch.optim.lr_scheduler.StepLR(optimizer_bilstm, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosine':
        scheduler_bilstm = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_bilstm, T_max=args.num_epochs, eta_min=1e-6)
    elif args.lr_scheduler == 'reducelronplateau':
        scheduler_bilstm = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_bilstm, mode='min', factor=args.lr_factor,
            patience=args.lr_patience, min_lr=1e-7
        )

    start_epoch_bilstm = 0
    if args.resume_checkpoint_bilstm and os.path.isfile(args.resume_checkpoint_bilstm):
        print(f"从检查点加载 BiLSTM+Attention 模型: {args.resume_checkpoint_bilstm}")
        checkpoint = torch.load(args.resume_checkpoint_bilstm, map_location=config.DEVICE)
        model_bilstm.load_state_dict(checkpoint['model_state_dict'])
        optimizer_bilstm.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler_bilstm is not None and 'scheduler_state_dict' in checkpoint:
            scheduler_bilstm.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch_bilstm = checkpoint['epoch']
        print(f"  模型权重和优化器状态已加载。将从 epoch {start_epoch_bilstm + 1} 继续训练。")

    checkpoint_dir_bilstm = os.path.join(config.CHECKPOINT_DIR, "bilstm_from_prepared_data")
    os.makedirs(checkpoint_dir_bilstm, exist_ok=True)

    train_losses_bilstm, val_losses_bilstm, model_bilstm_trained = train_model(
        model=model_bilstm,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_bilstm,
        criterion=criterion,
        num_epochs=args.num_epochs,
        device=config.DEVICE,
        model_name="BiLSTM_Attention",
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        checkpoint_dir=checkpoint_dir_bilstm,
        start_epoch=start_epoch_bilstm,
        scheduler=scheduler_bilstm
    )
    model_bilstm = model_bilstm_trained


    # --- 训练 CNN+BiLSTM ---
    model_cnnlstm = CNN_BiLSTM(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=num_classes,
        kernel_size=config.CNN_KERNEL_SIZE
    )
    model_cnnlstm.to(config.DEVICE)
    optimizer_cnnlstm = torch.optim.Adam(model_cnnlstm.parameters(), lr=args.learning_rate)

    # 添加学习率调度器
    scheduler_cnnlstm = None
    if args.lr_scheduler == 'step':
        scheduler_cnnlstm = torch.optim.lr_scheduler.StepLR(optimizer_cnnlstm, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosine':
        scheduler_cnnlstm = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_cnnlstm, T_max=args.num_epochs, eta_min=1e-6)
    elif args.lr_scheduler == 'reducelronplateau':
        scheduler_cnnlstm = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_cnnlstm, mode='min', factor=args.lr_factor,
            patience=args.lr_patience, min_lr=1e-7
        )

    start_epoch_cnnlstm = 0
    if args.resume_checkpoint_cnnlstm and os.path.isfile(args.resume_checkpoint_cnnlstm):
        print(f"从检查点加载 CNN_BiLSTM 模型: {args.resume_checkpoint_cnnlstm}")
        checkpoint = torch.load(args.resume_checkpoint_cnnlstm, map_location=config.DEVICE)
        model_cnnlstm.load_state_dict(checkpoint['model_state_dict'])
        optimizer_cnnlstm.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler_cnnlstm is not None and 'scheduler_state_dict' in checkpoint:
            scheduler_cnnlstm.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch_cnnlstm = checkpoint['epoch']
        print(f"  模型权重和优化器状态已加载。将从 epoch {start_epoch_cnnlstm + 1} 继续训练。")

    checkpoint_dir_cnnlstm = os.path.join(config.CHECKPOINT_DIR, "cnnlstm_from_prepared_data")
    os.makedirs(checkpoint_dir_cnnlstm, exist_ok=True)

    train_losses_cnnlstm, val_losses_cnnlstm, model_cnnlstm_trained = train_model(
        model=model_cnnlstm,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_cnnlstm,
        criterion=criterion,
        num_epochs=args.num_epochs,
        device=config.DEVICE,
        model_name="CNN_BiLSTM",
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        checkpoint_dir=checkpoint_dir_cnnlstm,
        start_epoch=start_epoch_cnnlstm,
        scheduler=scheduler_cnnlstm
    )
    model_cnnlstm = model_cnnlstm_trained

    # --- 步骤 5: 保存最终训练好的模型 ---
    print(f"\n--- 保存最终训练模型到 {config.MODEL_SAVE_PATH} (基于固定划分训练) ---")
    final_model_save_path = config.MODEL_SAVE_PATH.replace(".pth", "_from_prepared_data.pth")
    
    save_dict = {
        'mlb': mlb,
        'model_params': {
            'input_dim': config.INPUT_DIM,
            'hidden_dim': config.HIDDEN_DIM,
            'output_dim': num_classes,
            'num_lstm_layers': config.NUM_LSTM_LAYERS,
            'cnn_kernel_size': config.CNN_KERNEL_SIZE,
            'ensemble_weight_a': args.ensemble_weight_a
        },
        'training_args': vars(args)
    }
    
    if model_bilstm:
        save_dict['model_bilstm_state_dict'] = model_bilstm.state_dict()
    if model_cnnlstm:
        save_dict['model_cnnlstm_state_dict'] = model_cnnlstm.state_dict()

    # 创建并保存集成模型状态
    if model_bilstm and model_cnnlstm:
        ensemble_model = EnsembleModel(model_bilstm, model_cnnlstm, weightA=args.ensemble_weight_a)
        save_dict['ensemble_model_state_dict'] = ensemble_model.state_dict()

    torch.save(save_dict, final_model_save_path)
    print(f"模型和MLB已保存到 {final_model_save_path}")

    # 保存训练和验证损失
    np.save(os.path.join(config.OUTPUT_DIR, "bilstm_train_losses.npy"), np.array(train_losses_bilstm))
    np.save(os.path.join(config.OUTPUT_DIR, "bilstm_val_losses.npy"), np.array(val_losses_bilstm))
    np.save(os.path.join(config.OUTPUT_DIR, "cnnlstm_train_losses.npy"), np.array(train_losses_cnnlstm))
    np.save(os.path.join(config.OUTPUT_DIR, "cnnlstm_val_losses.npy"), np.array(val_losses_cnnlstm))

    print("\n--- 训练流程完成 ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于预处理数据进行模型训练的脚本")
    parser.add_argument('--prepared_data_dir', type=str, default=config.PREPARED_DATA_DIR, help='预处理和划分好的数据目录')
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--num_epochs', type=int, default=config.NUM_EPOCHS)
    parser.add_argument('--learning_rate', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--ensemble_weight_a', type=float, default=config.ENSEMBLE_WEIGHT_A)
    parser.add_argument('--early_stopping_patience', type=int, default=config.EARLY_STOPPING_PATIENCE)
    parser.add_argument('--early_stopping_min_delta', type=float, default=config.EARLY_STOPPING_MIN_DELTA)
    parser.add_argument('--resume_checkpoint_bilstm', type=str, default=config.RESUME_CHECKPOINT_BILSTM, help='BiLSTM检查点路径')
    parser.add_argument('--resume_checkpoint_cnnlstm', type=str, default=config.RESUME_CHECKPOINT_CNNLSTM, help='CNN_BiLSTM检查点路径')
    parser.add_argument('--lr_scheduler', type=str, default=config.LR_SCHEDULER, 
                        choices=['none', 'step', 'cosine', 'reducelronplateau'])
    parser.add_argument('--lr_step_size', type=int, default=config.LR_STEP_SIZE)
    parser.add_argument('--lr_gamma', type=float, default=config.LR_GAMMA)
    parser.add_argument('--lr_patience', type=int, default=config.LR_PATIENCE)
    parser.add_argument('--lr_factor', type=float, default=config.LR_FACTOR)


    cli_args = parser.parse_args()
    main(cli_args)