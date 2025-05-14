# evaluate_trained_model.py
import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc, precision_score, recall_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# --- 导入自定义模块 ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import config
from data_utils import (
    parse_uniprot_dat, load_data_from_excel, filter_annotations_by_category,
    build_go_to_slim_map, fast_map_to_slim, map_go_to_custom_classes,
    # encode_annotations, # 这里我们用加载的mlb
)
from feature_extractor import extract_protbert_features_batch
from datasets import ProteinFeatureDataset
from models import BiLSTMAttention, CNN_BiLSTM, EnsembleModel
from goatools.obo_parser import GODag # 需要用于数据预处理

def plot_roc_curve(fpr_dict, tpr_dict, roc_auc_dict, class_names, model_name="Model"):
    plt.figure(figsize=(10, 8))
    # Plot micro-average ROC curve
    if "micro" in fpr_dict:
        plt.plot(fpr_dict["micro"], tpr_dict["micro"],
                 label=f'Micro-average ROC curve (area = {roc_auc_dict["micro"]:.2f})',
                 color='deeppink', linestyle=':', linewidth=4)

    # Plot macro-average ROC curve
    if "macro" in fpr_dict:
        plt.plot(fpr_dict["macro"], tpr_dict["macro"],
                 label=f'Macro-average ROC curve (area = {roc_auc_dict["macro"]:.2f})',
                 color='navy', linestyle=':', linewidth=4)

    # Plot ROC curve for each class (optional, can be too cluttered for many classes)
    # from itertools import cycle
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    # for i, color in zip(range(len(class_names)), colors):
    #     if class_names[i] in fpr_dict: # Check if per-class roc was computed
    #         plt.plot(fpr_dict[class_names[i]], tpr_dict[class_names[i]], color=color, lw=2,
    #                  label=f'ROC curve of class {class_names[i]} (area = {roc_auc_dict[class_names[i]]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f"roc_curve_{model_name.replace(' ', '_')}.png")
    plt.show()

def plot_pr_curve(precision_dict, recall_dict, pr_auc_dict, class_names, model_name="Model"):
    plt.figure(figsize=(10, 8))
    if "micro" in precision_dict:
        plt.plot(recall_dict["micro"], precision_dict["micro"],
                 label=f'Micro-average PR curve (area = {pr_auc_dict["micro"]:.2f})',
                 color='gold', linestyle=':', linewidth=4)
    # Add macro average if computed and desired
    # ...
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.savefig(f"pr_curve_{model_name.replace(' ', '_')}.png")
    plt.show()


def evaluate_saved_model(model_path: str, test_data_file: str, args_from_training):
    print(f"--- 加载模型和评估测试集: {test_data_file} ---")
    print(f"使用模型文件: {model_path}")

    if not os.path.exists(model_path):
        print(f"错误: 模型文件未找到 {model_path}")
        return

    # 1. 加载检查点
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    mlb = checkpoint['mlb']
    ordered_ids = checkpoint.get('ordered_ids', None)  # 获取保存的ordered_ids
    model_params = checkpoint['model_params']
    training_args = checkpoint.get('training_args', {})  # 获取保存的训练参数
    num_classes = model_params['output_dim']

    # 使用保存的训练参数更新args_from_training
    for key, value in training_args.items():
        if hasattr(args_from_training, key):
            setattr(args_from_training, key, value)

    print(f"模型参数加载完成。类别数: {num_classes}")
    print(f"标签编码器 (mlb) 类别: {mlb.classes_}")

    # 2. 重新实例化模型
    model_bilstm = BiLSTMAttention(
        input_dim=model_params['input_dim'], hidden_dim=model_params['hidden_dim'],
        output_dim=num_classes, num_layers=model_params['num_lstm_layers']
    )
    model_cnnlstm = CNN_BiLSTM(
        input_dim=model_params['input_dim'], hidden_dim=model_params['hidden_dim'],
        output_dim=num_classes, kernel_size=model_params['cnn_kernel_size']
    )

    model_bilstm.load_state_dict(checkpoint['model_bilstm_state_dict'])
    model_cnnlstm.load_state_dict(checkpoint['model_cnnlstm_state_dict'])

    ensemble_model = EnsembleModel(model_bilstm, model_cnnlstm, weightA=model_params['ensemble_weight_a'])
    if 'ensemble_model_state_dict' in checkpoint:
         ensemble_model.load_state_dict(checkpoint['ensemble_model_state_dict'])

    models_to_evaluate = {
        "BiLSTM_Attention": model_bilstm,
        "CNN_BiLSTM": model_cnnlstm,
        "Ensemble_Model": ensemble_model
    }

    for model_name, model_instance in models_to_evaluate.items():
        model_instance.to(config.DEVICE)
        model_instance.eval()

    # 3. 准备测试数据
    print("\n--- 准备测试数据 ---")
    go_dag = GODag(config.OBO_FILE, optional_attrs={'relationship'})
    slim_terms = None
    if args_from_training.mapping_strategy == 'goslim':
        slim_dag = GODag(config.SLIM_OBO_FILE)
        slim_terms = set(slim_dag.keys())

    file_ext = os.path.splitext(test_data_file)[1].lower()
    if file_ext == '.dat':
        sequences, go_annotations, go_categories = parse_uniprot_dat(test_data_file)
    elif file_ext in ['.xlsx', '.xls']:
        sequences, go_annotations, go_categories = load_data_from_excel(test_data_file)
    else:
        print(f"错误: 不支持的测试数据文件格式 {file_ext}")
        return

    # 执行与训练时相同的筛选和映射
    all_prot_ids = set(sequences.keys())
    for prot_id in all_prot_ids:
        if prot_id not in go_categories:
            go_categories[prot_id] = {'MF': [], 'BP': [], 'CC': []}
        if prot_id not in go_annotations and prot_id in go_categories:
            all_ids_in_cats = set(go for cat_list in go_categories[prot_id].values() for go in cat_list)
            if all_ids_in_cats:
                go_annotations[prot_id] = list(all_ids_in_cats)

    category_annotations = filter_annotations_by_category(
        go_annotations, go_categories, args_from_training.target_go_category, go_dag
    )

    if args_from_training.mapping_strategy == 'goslim':
        go_to_slim_map = build_go_to_slim_map(go_dag, slim_terms)
        final_mapped_annotations = fast_map_to_slim(category_annotations, go_to_slim_map)
    elif args_from_training.mapping_strategy == 'custom':
        final_mapped_annotations = map_go_to_custom_classes(
            category_annotations, config.TARGET_MF_CLASSES, go_dag
        )
    else:
        print("错误：未知的映射策略")
        return

    # 使用加载的mlb直接转换标签
    test_protein_ids = []
    test_labels_original_format = []
    for pid, mapped_gos in final_mapped_annotations.items():
        if pid in sequences:
            test_protein_ids.append(pid)
            test_labels_original_format.append(mapped_gos)

    if not test_protein_ids:
        print("错误: 测试数据经过处理后没有剩余样本。")
        return

    try:
        test_encoded_labels = mlb.transform(test_labels_original_format).astype(np.float32)
    except ValueError as e:
        print(f"错误: 使用加载的MLB对象转换测试集标签时出错: {e}")
        print("这可能意味着测试集中的标签与训练时MLB学习到的类别不完全一致。")
        print(f"MLB类别: {mlb.classes_}")
        return

    # 提取特征
    test_sequences_dict = {pid: sequences[pid] for pid in test_protein_ids}
    test_features_dict = extract_protbert_features_batch(
        test_sequences_dict, config.CACHE_DIR, args_from_training.batch_size,
        config.MAX_SEQ_LENGTH, config.DEVICE
    )

    # 对齐特征和标签
    aligned_test_features = []
    aligned_test_labels = []
    final_test_ids = []
    for i, pid in enumerate(test_protein_ids):
        if pid in test_features_dict:
            aligned_test_features.append(test_features_dict[pid])
            aligned_test_labels.append(test_encoded_labels[i])
            final_test_ids.append(pid)

    if not final_test_ids:
        print("错误：测试特征提取或对齐后没有样本。")
        return

    test_encoded_labels_aligned = np.array(aligned_test_labels)
    test_dataset = ProteinFeatureDataset(final_test_ids, aligned_test_features, test_encoded_labels_aligned)
    test_loader = DataLoader(test_dataset, batch_size=args_from_training.batch_size, shuffle=False)

    print(f"测试数据准备完成。样本数: {len(test_dataset)}")

    # 4. 评估模型
    print("\n--- 开始评估模型 ---")
    results = {}
    for model_name, model in models_to_evaluate.items():
        print(f"\n评估 {model_name}...")
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"评估 {model_name}"):
                features = batch['features'].to(config.DEVICE)
                labels = batch['labels'].to(config.DEVICE)
                outputs = model(features)
                predictions = torch.sigmoid(outputs)
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)
        
        # 计算评估指标
        metrics = {}
        for i in range(num_classes):
            class_name = mlb.classes_[i]
            metrics[class_name] = {
                'precision': precision_score(all_labels[:, i], all_predictions[:, i] > 0.5),
                'recall': recall_score(all_labels[:, i], all_predictions[:, i] > 0.5),
                'f1': f1_score(all_labels[:, i], all_predictions[:, i] > 0.5),
                'auc': roc_auc_score(all_labels[:, i], all_predictions[:, i])
            }
        
        # 计算宏平均指标
        metrics['macro_avg'] = {
            'precision': precision_score(all_labels, all_predictions > 0.5, average='macro'),
            'recall': recall_score(all_labels, all_predictions > 0.5, average='macro'),
            'f1': f1_score(all_labels, all_predictions > 0.5, average='macro'),
            'auc': roc_auc_score(all_labels, all_predictions, average='macro')
        }
        
        # 计算微平均指标
        metrics['micro_avg'] = {
            'precision': precision_score(all_labels, all_predictions > 0.5, average='micro'),
            'recall': recall_score(all_labels, all_predictions > 0.5, average='micro'),
            'f1': f1_score(all_labels, all_predictions > 0.5, average='micro'),
            'auc': roc_auc_score(all_labels, all_predictions, average='micro')
        }
        
        results[model_name] = metrics
        
        # 打印评估结果
        print(f"\n{model_name} 评估结果:")
        print(f"宏平均指标:")
        print(f"  Precision: {metrics['macro_avg']['precision']:.4f}")
        print(f"  Recall: {metrics['macro_avg']['recall']:.4f}")
        print(f"  F1: {metrics['macro_avg']['f1']:.4f}")
        print(f"  AUC: {metrics['macro_avg']['auc']:.4f}")
        print(f"\n微平均指标:")
        print(f"  Precision: {metrics['micro_avg']['precision']:.4f}")
        print(f"  Recall: {metrics['micro_avg']['recall']:.4f}")
        print(f"  F1: {metrics['micro_avg']['f1']:.4f}")
        print(f"  AUC: {metrics['micro_avg']['auc']:.4f}")
    
    # 5. 保存评估结果
    results_dir = os.path.join(os.path.dirname(model_path), 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(results_dir, f'evaluation_results_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n评估结果已保存到: {results_file}")
    
    return results

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if np.isnan(obj): # 处理 NaN
            return None
        return super(NpEncoder, self).default(obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="加载训练好的模型并进行评估")
    parser.add_argument('--model_path', type=str, required=True, help='已训练模型的 .pth 文件路径')
    parser.add_argument('--test_data_file', type=str, required=True, help='测试数据文件路径 (.xlsx 或 .dat)')
    # 传递与训练时一致的数据处理参数
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
    parser.add_argument('--lr_scheduler', type=str, default=config.LR_SCHEDULER, choices=['none', 'step', 'cosine', 'reducelronplateau'], help='选择学习率调度器策略')
    parser.add_argument('--lr_step_size', type=int, default=config.LR_STEP_SIZE, help='StepLR 的 step_size')
    parser.add_argument('--lr_gamma', type=float, default=config.LR_GAMMA, help='StepLR/ExponentialLR 的 gamma')
    parser.add_argument('--lr_patience', type=int, default=config.LR_PATIENCE, help='ReduceLROnPlateau 的 patience')
    parser.add_argument('--lr_factor', type=float, default=config.LR_FACTOR, help='ReduceLROnPlateau 的 factor')


    cli_args = parser.parse_args()

    # 创建一个模拟的 args 对象，就像训练时使用的那样，包含数据处理的配置
    # 这样 evaluate_saved_model 可以使用与训练时一致的配置
    # 你可能需要从保存的 checkpoint['training_args'] 中加载这些参数
    # 这里我们简化为直接从命令行获取或使用默认值
    training_args_mock = argparse.Namespace(
        target_go_category=cli_args.target_go_category,
        mapping_strategy=cli_args.mapping_strategy,
        go_col_name=cli_args.go_col_name, # 确保这个与训练时一致
        batch_size=cli_args.batch_size
        # 其他在数据准备或特征提取中可能用到的参数...
    )

    evaluate_saved_model(cli_args.model_path, cli_args.test_data_file, training_args_mock)