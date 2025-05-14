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
import pickle
from typing import Tuple, List, Dict

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

def load_prepared_test_data(data_dir: str) -> Tuple[List[str], List[np.ndarray], np.ndarray, object]:
    """加载预处理好的测试数据"""
    ids = np.load(os.path.join(data_dir, "test_ids.npy"), allow_pickle=True).tolist()
    features_loaded = np.load(os.path.join(data_dir, "test_features.npy"), allow_pickle=True)
    features_list = [feat for feat in features_loaded]
    labels = np.load(os.path.join(data_dir, "test_labels.npy"), allow_pickle=True)
    
    mlb_path = os.path.join(data_dir, "mlb_encoder.pkl")
    if not os.path.exists(mlb_path):
        raise FileNotFoundError(f"MLB 编码器文件未找到于 {mlb_path}")
    with open(mlb_path, 'rb') as f:
        mlb = pickle.load(f)
    return ids, features_list, labels, mlb

def evaluate_on_fixed_test_set(model_checkpoint_path: str, prepared_data_dir: str, args_eval):
    """在固定的测试集上评估模型"""
    print(f"--- 在固定的测试集上评估模型 ---")
    print(f"使用模型文件: {model_checkpoint_path}")
    print(f"使用预处理数据目录: {prepared_data_dir}")

    if not os.path.exists(model_checkpoint_path):
        print(f"错误: 模型文件未找到 {model_checkpoint_path}")
        return
    if not os.path.exists(prepared_data_dir):
        print(f"错误: 预处理数据目录未找到 {prepared_data_dir}")
        return

    # 1. 加载模型检查点
    checkpoint = torch.load(model_checkpoint_path, map_location=config.DEVICE)
    model_params = checkpoint['model_params']
    num_classes_from_model = model_params['output_dim']

    # 2. 加载测试数据和MLB
    test_ids, test_features, test_labels, mlb = load_prepared_test_data(prepared_data_dir)
    
    if num_classes_from_model != len(mlb.classes_):
        print(f"警告: 模型输出维度 ({num_classes_from_model}) 与MLB类别数 ({len(mlb.classes_)}) 不匹配!")
    num_classes = len(mlb.classes_)

    # 3. 初始化模型
    model_bilstm = BiLSTMAttention(
        input_dim=model_params['input_dim'],
        hidden_dim=model_params['hidden_dim'],
        output_dim=num_classes,
        num_layers=model_params['num_lstm_layers']
    )
    model_cnnlstm = CNN_BiLSTM(
        input_dim=model_params['input_dim'],
        hidden_dim=model_params['hidden_dim'],
        output_dim=num_classes,
        kernel_size=model_params['cnn_kernel_size']
    )

    model_bilstm.load_state_dict(checkpoint['model_bilstm_state_dict'])
    model_cnnlstm.load_state_dict(checkpoint['model_cnnlstm_state_dict'])

    ensemble_model = EnsembleModel(
        model_bilstm,
        model_cnnlstm,
        weightA=model_params['ensemble_weight_a']
    )
    if 'ensemble_model_state_dict' in checkpoint:
        ensemble_model.load_state_dict(checkpoint['ensemble_model_state_dict'])

    # 4. 准备要评估的模型
    models_to_evaluate = {"Ensemble_Model": ensemble_model}
    if args_eval.eval_individual_models:
        models_to_evaluate["BiLSTM_Attention"] = model_bilstm
        models_to_evaluate["CNN_BiLSTM"] = model_cnnlstm

    for model_name, model in models_to_evaluate.items():
        model.to(config.DEVICE)
        model.eval()

    # 5. 创建测试数据加载器
    test_dataset = ProteinFeatureDataset(test_ids, test_features, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args_eval.batch_size_eval,
        shuffle=False,
        num_workers=0
    )

    # 6. 评估模型
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
                'precision': precision_score(all_labels[:, i], all_predictions[:, i] > args_eval.prediction_threshold),
                'recall': recall_score(all_labels[:, i], all_predictions[:, i] > args_eval.prediction_threshold),
                'f1': f1_score(all_labels[:, i], all_predictions[:, i] > args_eval.prediction_threshold),
                'auc': roc_auc_score(all_labels[:, i], all_predictions[:, i])
            }
        
        # 计算宏平均指标
        metrics['macro_avg'] = {
            'precision': precision_score(all_labels, all_predictions > args_eval.prediction_threshold, average='macro'),
            'recall': recall_score(all_labels, all_predictions > args_eval.prediction_threshold, average='macro'),
            'f1': f1_score(all_labels, all_predictions > args_eval.prediction_threshold, average='macro'),
            'auc': roc_auc_score(all_labels, all_predictions, average='macro')
        }
        
        # 计算微平均指标
        metrics['micro_avg'] = {
            'precision': precision_score(all_labels, all_predictions > args_eval.prediction_threshold, average='micro'),
            'recall': recall_score(all_labels, all_predictions > args_eval.prediction_threshold, average='micro'),
            'f1': f1_score(all_labels, all_predictions > args_eval.prediction_threshold, average='micro'),
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
    
    # 7. 保存评估结果
    results_dir = os.path.join(os.path.dirname(model_checkpoint_path), 'evaluation_results')
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
    parser = argparse.ArgumentParser(description="在固定的预处理测试集上评估已训练的模型")
    parser.add_argument('--model_checkpoint_path', type=str, required=True, help='已训练模型的.pth文件路径')
    parser.add_argument('--prepared_data_dir', type=str, required=True, help='包含预处理和划分好的数据的目录')
    parser.add_argument('--batch_size_eval', type=int, default=config.BATCH_SIZE, help='评估时的批次大小')
    parser.add_argument('--prediction_threshold', type=float, default=0.5, help='多标签预测的概率阈值')
    parser.add_argument('--eval_individual_models', action='store_true', help='是否也评估检查点中的单个基模型')

    cli_args_eval = parser.parse_args()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    evaluate_on_fixed_test_set(cli_args_eval.model_checkpoint_path, cli_args_eval.prepared_data_dir, cli_args_eval)