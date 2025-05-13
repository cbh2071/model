# -*- coding: utf-8 -*-
"""
使用 ProtBERT 提取蛋白质特征并进行缓存。
"""
import os
import hashlib
from typing import Dict, List

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer

# --- 从 config 导入 ---
from config import PROTBERT_MODEL_NAME, DEVICE # 使用 config 中定义的设备

# --- 全局加载 Tokenizer (只需一次) ---
# 放在全局避免在函数内反复加载
try:
    print(f"加载 ProtBERT Tokenizer: {PROTBERT_MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(PROTBERT_MODEL_NAME, do_lower_case=False)
except Exception as e:
    print(f"加载 ProtBERT Tokenizer 失败: {e}")
    tokenizer = None # 设置为 None 以便后续检查

# --- 特征提取函数 ---
def extract_protbert_features_batch(
    sequences: Dict[str, str],
    cache_dir: str,
    batch_size: int,
    max_length: int,
    device: torch.device = DEVICE # 允许覆盖全局设备
) -> Dict[str, np.ndarray]:
    """
    批量提取 ProtBERT 特征 (平均池化)，支持缓存。

    Args:
        sequences (Dict[str, str]): {protein_id: sequence}
        cache_dir (str): 缓存目录路径。
        batch_size (int): 处理批次大小。
        max_length (int): ProtBERT 最大序列长度。
        device (torch.device): 计算设备 (CPU or CUDA)。

    Returns:
        Dict[str, np.ndarray]: {protein_id: feature_vector}
    """
    if tokenizer is None:
        print("错误：ProtBERT Tokenizer 未成功加载，无法提取特征。")
        return {}

    print(f"开始提取 ProtBERT 特征，使用设备: {device}")
    os.makedirs(cache_dir, exist_ok=True)

    # --- 模型加载移到函数内部，确保在需要时加载到正确设备 ---
    model = None # 初始化为 None
    try:
        print("加载 ProtBERT 模型到设备...")
        model = BertModel.from_pretrained(PROTBERT_MODEL_NAME).to(device)
        model.eval()
        print("ProtBERT 模型加载完成。")
    except Exception as e:
        print(f"加载 ProtBERT 模型到设备 {device} 时出错: {e}")
        return {} # 无法加载模型则返回空

    all_features = {}
    sequences_to_process_ids = []
    sequences_to_process_list = []

    # 1. 检查缓存
    print("检查特征缓存...")
    protein_ids = list(sequences.keys())
    for prot_id in tqdm(protein_ids, desc="检查缓存"):
        seq = sequences[prot_id]
        cache_key = hashlib.md5(seq.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{cache_key}.npy")

        if os.path.exists(cache_path):
            try:
                feature = np.load(cache_path)
                all_features[prot_id] = feature
            except Exception as e:
                print(f"警告: 加载缓存文件 {cache_path} 失败: {e}. 将重新计算。")
                sequences_to_process_ids.append(prot_id)
                sequences_to_process_list.append(seq)
        else:
            sequences_to_process_ids.append(prot_id)
            sequences_to_process_list.append(seq)

    # 2. 处理需要计算的序列
    num_new_sequences = len(sequences_to_process_ids)
    if num_new_sequences > 0 and model is not None: # 确保模型已加载
        print(f"需要为 {num_new_sequences} 个序列计算新特征。")
        with torch.no_grad():
            for i in tqdm(range(0, num_new_sequences, batch_size), desc="提取特征"):
                batch_ids = sequences_to_process_ids[i : i + batch_size]
                batch_seqs_raw = sequences_to_process_list[i : i + batch_size]
                batch_seqs_processed = [" ".join(list(s)) for s in batch_seqs_raw]

                try:
                    inputs = tokenizer(
                        batch_seqs_processed,
                        add_special_tokens=True, padding="longest", truncation=True,
                        max_length=max_length, return_tensors="pt"
                    ).to(device)

                    outputs = model(**inputs)
                    hidden_states = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask']
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_hidden = torch.sum(hidden_states * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    mean_pooled_features = sum_hidden / sum_mask
                    batch_features_np = mean_pooled_features.cpu().numpy()

                    for j, prot_id in enumerate(batch_ids):
                        feature = batch_features_np[j]
                        all_features[prot_id] = feature
                        seq = sequences[prot_id]
                        cache_key = hashlib.md5(seq.encode()).hexdigest()
                        cache_path = os.path.join(cache_dir, f"{cache_key}.npy")
                        try:
                            np.save(cache_path, feature)
                        except Exception as e:
                            print(f"警告: 保存缓存文件 {cache_path} 失败: {e}")

                except RuntimeError as e:
                     if "out of memory" in str(e):
                         print(f"\nGPU 显存不足！在处理批次 {i//batch_size + 1} 时发生。尝试减小 BATCH_SIZE。")
                         # 可以选择在这里清理缓存并退出，或者继续尝试（可能失败）
                         del model # 尝试释放模型占用的显存
                         if torch.cuda.is_available(): torch.cuda.empty_cache()
                         return {} # 返回空字典表示失败
                         # print("跳过当前批次...")
                         # continue
                     else:
                         print(f"\n处理批次 {i//batch_size + 1} 时发生运行时错误: {e}")
                         continue
                except Exception as e:
                    print(f"\n处理批次 {i//batch_size + 1} 时发生未知错误: {e}")
                    continue

        # --- 在函数结束前清理模型占用的显存 ---
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elif num_new_sequences > 0 and model is None:
        print("错误：模型未能加载，无法计算新特征。")


    print(f"特征提取完成。共获得 {len(all_features)} 个蛋白质的特征。")
    return all_features