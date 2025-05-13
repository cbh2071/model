# -*- coding: utf-8 -*-
"""
主执行脚本：蛋白质功能预测流程编排。
"""
import argparse
import os
import sys
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from goatools.obo_parser import GODag

# --- 导入自定义模块 ---
# 添加项目根目录到 Python 路径，以便导入模块
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import config # 导入配置
from data_utils import ( # 导入数据处理函数
    parse_uniprot_dat, load_data_from_excel, filter_annotations_by_category,
    build_go_to_slim_map, fast_map_to_slim, map_go_to_custom_classes,
    encode_annotations, save_label_distribution, diagnose_mapping
)
from feature_extractor import extract_protbert_features_batch # 导入特征提取器
from datasets import ProteinFeatureDataset # 导入 Dataset
from models import BiLSTMAttention, CNN_BiLSTM, EnsembleModel # 导入模型
from training_utils import train_model, evaluate_model # 导入训练/评估工具

def main(args):
    """主执行流程"""
    print("--- 蛋白质功能预测任务启动 ---")
    print(f"使用设备: {config.DEVICE}")

    # --- 步骤 1: 加载 GO 本体和 Slim 文件 ---
    print("\n--- 步骤 1: 加载 GO 本体文件 ---")
    try:
        go_dag = GODag(config.OBO_FILE, optional_attrs={'relationship'})
        if args.mapping_strategy == 'goslim':
            slim_dag = GODag(config.SLIM_OBO_FILE)
            slim_terms = set(slim_dag.keys())
            print(f"Slim OBO 文件加载成功 ({len(slim_terms)} terms)。")
        print(f"基础 OBO 文件加载成功 ({len(go_dag)} terms)。")
    except FileNotFoundError as e:
        print(f"错误: OBO 文件未找到: {e}")
        return
    except Exception as e:
        print(f"加载 OBO 文件时出错: {e}")
        return

    # --- 步骤 2: 加载输入数据 ---
    print(f"\n--- 步骤 2: 加载输入数据 ({args.input_data_file}) ---")
    file_ext = os.path.splitext(args.input_data_file)[1].lower()
    sequences: Dict[str, str]
    go_annotations: Dict[str, List[str]]
    go_categories: Dict[str, Dict[str, List[str]]]

    if file_ext == '.dat':
        sequences, go_annotations, go_categories = parse_uniprot_dat(args.input_data_file)
    elif file_ext in ['.xlsx', '.xls']:
        sequences, go_annotations, go_categories = load_data_from_excel(
            args.input_data_file
            # 可根据需要传递 id_col, seq_col, go_col 参数
        )
    else:
        print(f"错误: 不支持的文件格式 '{file_ext}'。请提供 .dat 或 .xlsx 文件。")
        return

    if not sequences:
        print("错误: 未能从输入文件加载任何序列数据。")
        return

    # --- 步骤 3: 按类别筛选注释 ---
    print(f"\n--- 步骤 3: 按类别 '{args.target_go_category}' 筛选注释 ---")
    # 确保 go_categories 覆盖所有序列 ID
    all_prot_ids = set(sequences.keys())
    for prot_id in all_prot_ids:
        if prot_id not in go_categories:
            go_categories[prot_id] = {'MF': [], 'BP': [], 'CC': []}
        # 补充 go_annotations (如果 Excel 加载时没提取)
        if prot_id not in go_annotations and prot_id in go_categories:
            all_ids_in_cats = set(go for cat_list in go_categories[prot_id].values() for go in cat_list)
            if all_ids_in_cats:
                 go_annotations[prot_id] = list(all_ids_in_cats)

    category_annotations = filter_annotations_by_category(
        go_annotations, # 传入所有GO注释
        go_categories,
        args.target_go_category,
        go_dag # 传入 OBO 文件用于推断
    )

    valid_protein_ids_cat = set(category_annotations.keys())
    print(f"筛选后，{len(valid_protein_ids_cat)} 个蛋白质具有 '{args.target_go_category}' 注释。")
    if not valid_protein_ids_cat:
        print("错误: 筛选后没有蛋白质剩下。")
        return

    # --- 步骤 4: GO Slim 或 自定义 映射 ---
    print(f"\n--- 步骤 4: 使用 '{args.mapping_strategy}' 策略进行标签映射 ---")
    final_mapped_annotations: Dict[str, List[str]]
    if args.mapping_strategy == 'goslim':
        go_to_slim_map = build_go_to_slim_map(go_dag, slim_terms)
        final_mapped_annotations = fast_map_to_slim(category_annotations, go_to_slim_map)
    elif args.mapping_strategy == 'custom':
        final_mapped_annotations = map_go_to_custom_classes(
            category_annotations,
            config.TARGET_MF_CLASSES, # 从 config 获取
            go_dag,
        )
    else: # Should not happen due to argparse choices
        print(f"错误: 未知的映射策略 '{args.mapping_strategy}'")
        return

    valid_protein_ids_mapped = set(final_mapped_annotations.keys())
    print(f"映射后，{len(valid_protein_ids_mapped)} 个蛋白质至少有一个最终类别标签。")

    final_protein_ids = list(valid_protein_ids_mapped)
    if not final_protein_ids:
        print("错误: 没有蛋白质在映射后保留下来。无法继续。")
        return

    final_sequences_dict = {pid: sequences[pid] for pid in final_protein_ids if pid in sequences}
    print(f"最终用于模型训练的蛋白质数量: {len(final_protein_ids)}")

    # --- (可选) 诊断映射结果 ---
    if args.diagnose:
        diagnose_mapping(category_annotations, final_mapped_annotations, go_dag, num_samples=5)

    # --- 步骤 5: 编码标签 ---
    print("\n--- 步骤 5: 编码最终标签 ---")
    ordered_ids, encoded_labels, mlb = encode_annotations(final_mapped_annotations)

    if encoded_labels.size == 0:
        print("错误: 标签编码失败。")
        return

    num_classes = encoded_labels.shape[1]
    print(f"标签编码完成，类别数: {num_classes} ({len(mlb.classes_)} unique labels)")
    print(f"类别列表: {mlb.classes_}")

    # 保存标签分布
    dist_file = os.path.join(config.DISTRIBUTION_FILE_PREFIX, f"{args.mapping_strategy}_label_distribution.txt")
    save_label_distribution(encoded_labels, mlb, output_file=dist_file)

    # --- 步骤 6: 提取/加载 ProtBERT 特征 ---
    print("\n--- 步骤 6: 提取/加载 ProtBERT 特征 ---")
    final_sequences_ordered_dict = {pid: final_sequences_dict[pid] for pid in ordered_ids}
    features_dict = extract_protbert_features_batch(
        final_sequences_ordered_dict,
        cache_dir=config.CACHE_DIR,
        batch_size=args.batch_size, # 使用命令行传入的 batch size
        max_length=config.MAX_SEQ_LENGTH,
        device=config.DEVICE
    )

    if len(features_dict) != len(ordered_ids):
        print(f"警告: 特征提取/加载后的蛋白质数量 ({len(features_dict)}) 与标签数量 ({len(ordered_ids)}) 不匹配。进行对齐...")
        valid_ids_after_feature = list(set(ordered_ids).intersection(features_dict.keys()))
        if not valid_ids_after_feature:
            print("错误：特征提取后没有有效的蛋白质。")
            return
        id_to_index = {pid: i for i, pid in enumerate(ordered_ids)}
        indices_to_keep = [id_to_index[pid] for pid in valid_ids_after_feature]
        encoded_labels = encoded_labels[indices_to_keep]
        ordered_ids = valid_ids_after_feature # 更新 ID 列表
        print(f"对齐后，有效蛋白质数量：{len(ordered_ids)}")
    # 确保特征列表顺序与更新后的 ordered_ids 一致
    features_list = [features_dict[pid] for pid in ordered_ids]


    # --- 步骤 7: 准备数据集和数据加载器 ---
    print("\n--- 步骤 7: 准备数据集和数据加载器 ---")
    num_samples = len(ordered_ids)
    indices = np.arange(num_samples)

    # 尝试分层划分 (如果标签是 one-hot 或单标签形式)
    # 对于多标签，简单的argmax分层可能不是最优，但可以尝试
    stratify_labels = None
    if num_samples > 1 and len(encoded_labels.shape) > 1 and encoded_labels.shape[1] > 0 :
        # 尝试使用标签组合作为分层依据 (如果组合数不多)
        try:
             label_tuples = [tuple(row) for row in encoded_labels]
             label_counts = Counter(label_tuples)
             # 只对出现次数超过 1 次的标签组合进行分层
             if len(label_counts) < num_samples / 2: # 启发式：组合数不过多
                 stratify_labels = label_tuples
             else: # 否则退化为按第一个标签分层
                 print("标签组合过多，尝试按第一个标签分层...")
                 stratify_labels = encoded_labels.argmax(1)
        except Exception as e:
             print(f"生成分层标签时出错: {e}. 使用非分层划分。")
             stratify_labels = None
    elif num_samples > 1 and len(encoded_labels.shape) == 1: # 单标签情况
         stratify_labels = encoded_labels

    try:
        train_val_idx, test_idx = train_test_split(
            indices, test_size=args.test_size, random_state=42,
            stratify=stratify_labels
        )
        # 为验证集再次计算分层标签
        stratify_val = None
        if stratify_labels is not None:
            stratify_val = [stratify_labels[i] for i in train_val_idx]

        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=args.val_size / (1 - args.test_size), random_state=42,
            stratify=stratify_val
        )
    except ValueError as e:
         print(f"警告: 分层划分失败 ({e})。将使用非分层划分。")
         train_val_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=42)
         train_idx, val_idx = train_test_split(train_val_idx, test_size=args.val_size / (1 - args.test_size), random_state=42)

    train_features = [features_list[i] for i in train_idx]
    train_labels = encoded_labels[train_idx]
    train_ids = [ordered_ids[i] for i in train_idx]
    train_dataset = ProteinFeatureDataset(train_ids, train_features, train_labels)

    val_features = [features_list[i] for i in val_idx]
    val_labels = encoded_labels[val_idx]
    val_ids = [ordered_ids[i] for i in val_idx]
    val_dataset = ProteinFeatureDataset(val_ids, val_features, val_labels)

    test_features = [features_list[i] for i in test_idx]
    test_labels = encoded_labels[test_idx]
    test_ids = [ordered_ids[i] for i in test_idx]
    test_dataset = ProteinFeatureDataset(test_ids, test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"数据集划分: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}, 测试集 {len(test_dataset)}")

    # --- 步骤 8: 模型初始化与训练 ---
    print("\n--- 步骤 8: 模型初始化与训练 ---")
    criterion = nn.BCEWithLogitsLoss() # 多标签分类损失

    # --- 训练 BiLSTM+Attention ---
    model_bilstm = BiLSTMAttention(
        input_dim=config.INPUT_DIM, hidden_dim=config.HIDDEN_DIM,
        output_dim=num_classes, num_layers=config.NUM_LSTM_LAYERS
    )
    model_bilstm.to(config.DEVICE) # 先移动到设备
    optimizer_bilstm = torch.optim.Adam(model_bilstm.parameters(), lr=args.learning_rate)
    _, _, model_bilstm = train_model(
        model=model_bilstm, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer_bilstm, criterion=criterion, num_epochs=args.num_epochs,
        device=config.DEVICE, model_name="BiLSTM_Attention",
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta
    )

    start_epoch_bilstm = 0
    if args.resume_checkpoint and "BiLSTM_Attention" in args.resume_checkpoint: # 简单判断是否是该模型的检查点
        if os.path.isfile(args.resume_checkpoint):
            print(f"从检查点加载 BiLSTM+Attention 模型: {args.resume_checkpoint}")
            checkpoint = torch.load(args.resume_checkpoint, map_location=config.DEVICE) # 加载到指定设备
            model_bilstm.load_state_dict(checkpoint['model_state_dict'])
            optimizer_bilstm.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch_bilstm = checkpoint['epoch'] # 下一个epoch从这里开始
            # best_val_loss_bilstm = checkpoint.get('best_val_loss', float('inf')) # 可选：恢复最佳损失记录
            print(f"  模型权重和优化器状态已加载。将从 epoch {start_epoch_bilstm + 1} 继续训练。")
        else:
            print(f"警告: 指定的检查点文件未找到: {args.resume_checkpoint}")

    _, _, model_bilstm = train_model(
        model=model_bilstm, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer_bilstm, criterion=criterion,
        num_epochs=args.num_epochs, # 总 epoch 数不变
        device=config.DEVICE, model_name="BiLSTM_Attention",
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        checkpoint_dir=os.path.join(config.CHECKPOINT_DIR_BASE, "bilstm"), # 为不同模型指定不同检查点子目录
        start_epoch=start_epoch_bilstm # 传递起始 epoch
    )

    # --- 训练 CNN+BiLSTM ---
    model_cnnlstm = CNN_BiLSTM(
        input_dim=config.INPUT_DIM, hidden_dim=config.HIDDEN_DIM, output_dim=num_classes,
        kernel_size=config.CNN_KERNEL_SIZE, num_layers=1 # 通常 CNN 后 LSTM 层数不需要太多
    )
    model_cnnlstm.to(config.DEVICE) # 先移动到设备
    optimizer_cnnlstm = torch.optim.Adam(model_cnnlstm.parameters(), lr=args.learning_rate)
    _, _, model_cnnlstm = train_model(
        model=model_cnnlstm, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer_cnnlstm, criterion=criterion, num_epochs=args.num_epochs,
        device=config.DEVICE, model_name="CNN_BiLSTM",
        patience=args.early_stopping_patience, # 从命令行参数获取
        min_delta=args.early_stopping_min_delta # 从命令行参数获取
    )

    start_epoch_cnnlstm = 0
    if args.resume_checkpoint and "CNN_BiLSTM" in args.resume_checkpoint:
        if os.path.isfile(args.resume_checkpoint):
            print(f"从检查点加载 CNN_BiLSTM 模型: {args.resume_checkpoint}")
            checkpoint = torch.load(args.resume_checkpoint, map_location=config.DEVICE)
            model_cnnlstm.load_state_dict(checkpoint['model_state_dict'])
            optimizer_cnnlstm.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch_cnnlstm = checkpoint['epoch']
            print(f"  模型权重和优化器状态已加载。将从 epoch {start_epoch_cnnlstm + 1} 继续训练。")
        else:
            print(f"警告: 指定的检查点文件未找到: {args.resume_checkpoint}")

    _, _, model_cnnlstm = train_model(
        model=model_cnnlstm, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer_cnnlstm, criterion=criterion,
        num_epochs=args.num_epochs,
        device=config.DEVICE, model_name="CNN_BiLSTM",
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        checkpoint_dir=os.path.join(config.CHECKPOINT_DIR_BASE, "cnnlstm"),
        start_epoch=start_epoch_cnnlstm
    )

    # --- 步骤 9: 集成与评估 ---
    print("\n--- 步骤 9: 集成模型与最终评估 ---")
    ensemble_model = EnsembleModel(model_bilstm, model_cnnlstm, weightA=args.ensemble_weight_a)
    ensemble_model.to(config.DEVICE)

    test_loss, sample_f1, micro_f1, weighted_f1, class_f1_dict = evaluate_model(
        model=ensemble_model, test_loader=test_loader, criterion=criterion,
        device=config.DEVICE, mlb=mlb
    )

    print("\n--- 最终集成模型测试结果 ---")
    print(f"  - 平均测试损失: {test_loss:.4f}")
    print(f"  - Sample-based F1: {sample_f1:.4f}")
    print(f"  - Micro F1: {micro_f1:.4f}")
    print(f"  - Weighted F1: {weighted_f1:.4f}")

    # --- 步骤 10: 保存模型 ---
    print(f"\n--- 步骤 10: 保存模型到 {config.MODEL_SAVE_PATH} ---")
    try:
        torch.save({
            'model_bilstm_state_dict': model_bilstm.state_dict(),
            'model_cnnlstm_state_dict': model_cnnlstm.state_dict(),
            'ensemble_model_state_dict': ensemble_model.state_dict(),
            'mlb': mlb,
            'model_params': {
                'input_dim': config.INPUT_DIM, 'hidden_dim': config.HIDDEN_DIM,
                'output_dim': num_classes, 'num_lstm_layers': config.NUM_LSTM_LAYERS,
                'cnn_kernel_size': config.CNN_KERNEL_SIZE,
                'ensemble_weight_a': args.ensemble_weight_a
            },
            'mapping_strategy': args.mapping_strategy,
            'target_go_category': args.target_go_category,
        }, config.MODEL_SAVE_PATH)
        print("模型和标签编码器保存成功。")
    except Exception as e:
        print(f"保存模型时出错: {e}")

    print("\n--- 任务完成 ---")


# --- 主程序入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="蛋白质功能预测模型训练脚本")

    parser.add_argument('--input_data_file', type=str, default=config.INPUT_DATA_FILE, help='输入数据文件路径 (.xlsx 或 .dat)')
    parser.add_argument('--target_go_category', type=str, default=config.TARGET_GO_CATEGORY, choices=['MF', 'BP', 'CC'], help='要预测的 GO 类别 (default: MF)')
    parser.add_argument('--mapping_strategy', type=str, default=config.MAPPING_STRATEGY, choices=['goslim', 'custom'], help='标签映射策略 (default: goslim)')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help=f'批次大小 (default: {config.BATCH_SIZE})')
    parser.add_argument('--num_epochs', type=int, default=config.NUM_EPOCHS, help=f'训练轮数 (default: {config.NUM_EPOCHS})')
    parser.add_argument('--learning_rate', type=float, default=config.LEARNING_RATE, help=f'学习率 (default: {config.LEARNING_RATE})')
    parser.add_argument('--test_size', type=float, default=config.TEST_SIZE, help=f'测试集比例 (default: {config.TEST_SIZE})')
    parser.add_argument('--val_size', type=float, default=config.VAL_SIZE, help=f'验证集比例 (default: {config.VAL_SIZE})')
    parser.add_argument('--ensemble_weight_a', type=float, default=config.ENSEMBLE_WEIGHT_A, help=f'集成模型中模型A(BiLSTM)的权重 (default: {config.ENSEMBLE_WEIGHT_A})')
    parser.add_argument('--diagnose', action='store_true', help='是否执行映射诊断步骤')
    parser.add_argument('--early_stopping_patience', type=int, default=config.EARLY_STOPPING_PATIENCE, help=f'早停策略的耐心值 (default: {config.EARLY_STOPPING_PATIENCE})')
    parser.add_argument('--early_stopping_min_delta', type=float, default=config.EARLY_STOPPING_MIN_DELTA, help=f'早停策略的最小变化值 (default: {config.EARLY_STOPPING_MIN_DELTA})')
    parser.add_argument('--resume_checkpoint', type=str, default=config.CHECKPOINT_DIR, help=f'检查点保存目录 (default: {config.CHECKPOINT_DIR})')

    args = parser.parse_args()

    # 检查 OBO 文件是否存在
    if not os.path.exists(config.OBO_FILE):
        print(f"错误: GO OBO 文件未找到于: {config.OBO_FILE}")
        sys.exit(1)
    if args.mapping_strategy == 'goslim' and not os.path.exists(config.SLIM_OBO_FILE):
        print(f"错误: GO Slim OBO 文件未找到于: {config.SLIM_OBO_FILE} (当 mapping_strategy 为 goslim 时需要)")
        sys.exit(1)

    main(args)