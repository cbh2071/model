# -*- coding: utf-8 -*-
"""
数据处理工具函数：
- 加载 .dat, .xlsx 数据
- GO 类别筛选
- GO Slim / 自定义类别映射
- 标签编码
- 标签分布统计与保存
- 映射诊断
"""
import re
import os
from typing import Dict, List, Tuple, Set, Optional, Union
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from Bio import SeqIO
from goatools.obo_parser import GODag
from goatools.godag.go_tasks import get_go2parents # 修正导入
from sklearn.preprocessing import MultiLabelBinarizer

# --- 从 config 导入 ---
# (如果需要可以在函数参数中传递，或者直接导入)
# from config import TARGET_MF_CLASSES

# --- 数据加载函数 ---

def parse_uniprot_dat(file_path: str) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
    """解析 UniProtKB/Swiss-Prot 格式的 .dat 文件。"""
    print(f"从 .dat 文件加载数据: {file_path}")
    sequences = {}
    go_annotations = defaultdict(list) # 使用 defaultdict
    go_categories = defaultdict(lambda: {'MF': [], 'BP': [], 'CC': []}) # 使用 defaultdict

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 分割条目，更稳健的方式
            entry_content = ""
            for line in f:
                entry_content += line
                if line.startswith('//'):
                    if entry_content.strip(): # 处理非空条目
                        entry = entry_content
                        # ... (解析逻辑与之前相同) ...
                        ac_match = re.search(r'^AC\s+([^;]+);', entry, re.MULTILINE)
                        protein_id = ac_match.group(1).strip() if ac_match else None
                        if not protein_id:
                            entry_content = "" # 重置内容
                            continue

                        seq_lines = re.findall(r'^SQ.*?\n((?:^[ \t]+.*\n?)+)', entry, re.MULTILINE | re.DOTALL)
                        sequence = ''
                        if seq_lines:
                            sequence = ''.join(seq_lines[0].split()).upper()
                        sequences[protein_id] = sequence

                        go_matches = re.findall(r'^DR\s+GO;\s+(GO:\d+);\s+([CPF]):', entry, re.MULTILINE)
                        current_go_ids = []
                        current_go_cats = {'MF': [], 'BP': [], 'CC': []}
                        for go_id, cat_code in go_matches:
                            current_go_ids.append(go_id)
                            category = {'F': 'MF', 'P': 'BP', 'C': 'CC'}.get(cat_code)
                            if category:
                                current_go_cats[category].append(go_id)

                        if current_go_ids:
                             go_annotations[protein_id] = current_go_ids # 直接赋值给defaultdict
                        # 总是记录分类信息，即使 GO 注释列表可能为空 (如果只找到 BP/CC 但未找到 MF)
                        go_categories[protein_id] = current_go_cats

                    entry_content = "" # 重置以处理下一个条目
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
        return {}, {}, {}
    except Exception as e:
        print(f"读取 .dat 文件时出错: {e}")
        return {}, {}, {}

    print(f"从 .dat 文件加载了 {len(sequences)} 个序列，{len(go_annotations)} 个条目具有 GO 注释。")
    return dict(sequences), dict(go_annotations), dict(go_categories)


def load_data_from_excel(
    file_path: str,
    id_col: str = 'Entry',
    seq_col: str = 'Sequence',
    # --- 修改点 1: 更新默认的 GO 列名 ---
    go_col: str = 'Gene Ontology (molecular function)',
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
    """
    从 Excel 文件加载蛋白质序列和特定类别的 GO 注释。
    假定 go_col 参数指向的列只包含特定本体（如 MF）的 GO terms。

    Args:
        file_path (str): Excel 文件路径。
        id_col (str): 蛋白质 ID 列名。
        seq_col (str): 蛋白质序列列名。
        go_col (str): 包含特定类别 (如 MF) GO 注释的列名。

    Returns:
        Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
            - sequences (dict): {protein_id: sequence}
            - go_annotations (dict): {protein_id: [go_id1, go_id2, ...]} (仅来自 go_col)
            - go_categories (dict): {protein_id: {'MF': [...], 'BP': [], 'CC': []}} (直接将 go_col 的注释归类为 MF)
    """
    print(f"从 Excel 文件加载数据: {file_path}")
    print(f"  ID 列: '{id_col}', 序列列: '{seq_col}', GO(MF)列: '{go_col}'") # 确认列名
    sequences = {}
    # go_annotations 仍然可以收集所有找到的 GO ID
    go_annotations = defaultdict(list)
    # go_categories 初始化，准备将 go_col 的 ID 放入 'MF'
    go_categories = defaultdict(lambda: {'MF': [], 'BP': [], 'CC': []})

    try:
        df = pd.read_excel(file_path, dtype={id_col: str, seq_col: str, go_col: str})
        # --- 修改点 2: 确保填充 go_col 的 NaN ---
        df.fillna({go_col: ''}, inplace=True)
    except FileNotFoundError:
        print(f"错误: Excel 文件未找到 {file_path}")
        return {}, {}, {}
    except ValueError as e:
        # 捕获列不存在的错误
        if f"'{go_col}'" in str(e):
             print(f"错误: Excel 文件中未找到指定的 GO 列 '{go_col}'。请检查列名或文件。")
             print(f"可用列: {df.columns.tolist()}")
             return {}, {}, {}
        else:
            print(f"读取 Excel 文件时发生值错误: {e}")
            return {}, {}, {}
    except Exception as e:
        print(f"读取 Excel 文件时发生其他错误: {e}")
        return {}, {}, {}

    required_cols = [id_col, seq_col, go_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        # 这个检查理论上会被上面的 try-except 捕获，但保留以防万一
        print(f"错误: Excel 文件中缺少必需的列: {', '.join(missing_cols)}")
        return {}, {}, {}

    go_id_pattern = re.compile(r"GO:\d{7}")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="处理 Excel 行"):
        protein_id = row[id_col].strip()
        sequence = row[seq_col].strip().upper()
        go_terms_str = row[go_col] # 已经是字符串

        if not protein_id or not sequence:
            continue # 跳过无效行

        # 验证氨基酸序列 (保持)
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        if not set(sequence).issubset(valid_aa):
            # print(f"警告: 蛋白质 {protein_id} 序列包含非标准氨基酸，已跳过。")
            continue

        sequences[protein_id] = sequence

        # 从指定的 go_col 中提取所有 GO ID
        found_go_ids = go_id_pattern.findall(go_terms_str)
        if found_go_ids:
            unique_go_ids = sorted(list(set(found_go_ids))) # 排序并去重
            # 存储所有找到的 GO ID
            go_annotations[protein_id] = unique_go_ids

            # --- 修改点 3: 直接将这些 ID 归类为 MF ---
            # 不需要再进行关键词猜测
            go_categories[protein_id]['MF'] = unique_go_ids
            # BP 和 CC 保持为空列表 (defaultdict 会处理)

    print(f"从 Excel 文件加载了 {len(sequences)} 个序列。")
    print(f"在列 '{go_col}' 中找到 {len(go_annotations)} 个条目具有 GO 注释，已全部归类为 MF。")

    return dict(sequences), dict(go_annotations), dict(go_categories)


# --- GO 类别筛选 ---
def filter_annotations_by_category(
    go_annotations: Dict[str, List[str]], # 输入改为所有GO注释
    go_categories: Dict[str, Dict[str, List[str]]],
    target_category: str,
    go_dag: Optional[GODag] = None
) -> Dict[str, List[str]]:
    """根据指定的 GO 类别 (MF, BP, CC) 筛选注释。"""
    print(f"按类别 '{target_category}' 筛选 GO 注释...")
    filtered_annotations = {}
    needs_inference = False

    if go_dag is None:
        print("警告: 未提供 GODag 对象，将仅依赖 go_categories 中的信息进行筛选。")
    else:
        needs_inference = True # 总是允许使用 OBO 进行补充或验证
        print("将使用 GODag 辅助进行类别验证和推断。")

    category_map = {'MF': 'molecular_function', 'BP': 'biological_process', 'CC': 'cellular_component'}
    target_namespace = category_map.get(target_category)
    if not target_namespace:
        print(f"错误: 无效的目标类别 '{target_category}'。")
        return {}

    for protein_id, original_go_ids in tqdm(go_annotations.items(), desc="筛选类别"):
        category_go_ids = set() # 使用集合去重

        # 1. 从 go_categories 获取该类别的 ID
        if protein_id in go_categories and target_category in go_categories[protein_id]:
            category_go_ids.update(go_categories[protein_id][target_category])

        # 2. 如果需要，使用 OBO 文件进行验证或推断
        if needs_inference:
            for go_id in original_go_ids:
                 term = go_dag.get(go_id)
                 if term and term.namespace == target_namespace:
                     category_go_ids.add(go_id) # 确保所有属于该命名空间的都被加入

        if category_go_ids:
            filtered_annotations[protein_id] = sorted(list(category_go_ids))

    print(f"筛选完成，{len(filtered_annotations)} 个蛋白质具有 '{target_category}' 类别的注释。")
    return filtered_annotations


# --- GO Slim 映射函数 ---
def build_go_to_slim_map(go_dag: GODag, slim_terms: Set[str]) -> Dict[str, Set[str]]:
    """预构建所有 GO Term 到其对应 GO Slim Term 的映射字典。"""
    # (代码与之前版本相同)
    print("构建 GO 到 GO Slim 的映射...")
    go_to_slim = {}
    all_go_terms = set(go_dag.keys())

    for go_id in tqdm(all_go_terms, desc="处理 GO Terms"):
        if go_id not in go_dag: # 安全检查
             continue
        node = go_dag[go_id]
        # 查找所有祖先（包括自身）中的 slim terms
        slim_ancestors = get_all_parents_of_term(go_id, go_dag, include_self=True)
        if node.id in slim_terms: # 添加自身（如果自身是 slim）
             slim_ancestors.add(node.id)

        if slim_ancestors:
            go_to_slim[go_id] = slim_ancestors

    # 确保 Slim Term 自身映射到自身
    for slim_id in slim_terms:
        if slim_id in go_dag:
            if slim_id not in go_to_slim:
                go_to_slim[slim_id] = {slim_id}
            else:
                go_to_slim[slim_id].add(slim_id)

    print(f"GO 到 GO Slim 映射构建完成，覆盖 {len(go_to_slim)} 个 GO terms。")
    return go_to_slim

def fast_map_to_slim(
    annotation_dict: Dict[str, List[str]],
    go_to_slim_map: Dict[str, Set[str]]
) -> Dict[str, List[str]]:
    """利用预构建的映射字典，将原始 GO 注释快速转换为 GO Slim 注释。"""
    print("将注释映射到 GO Slim...")
    new_annotations = {}
    missing_map_count = 0
    for prot_id, go_ids in tqdm(annotation_dict.items(), desc="映射蛋白质注释"):
        mapped_terms = set()
        for go_id in go_ids:
            if go_id in go_to_slim_map:
                mapped_terms.update(go_to_slim_map[go_id])
            else: # 可选：统计有多少原始GO ID没有映射到Slim
                missing_map_count += 1
        if mapped_terms:
            new_annotations[prot_id] = sorted(list(mapped_terms))
    if missing_map_count > 0:
        print(f"  警告: {missing_map_count} 个原始 GO ID 未在映射表中找到对应的 Slim Term。")
    print(f"映射完成，{len(new_annotations)} 个蛋白质至少映射到一个 Slim Term。")
    return new_annotations


# --- 自定义类别映射函数 ---
def get_all_parents_of_term(go_id: str, obo_dag: Dict[str, Set[str]], include_self: bool = False) -> Set[str]:
    """
    获取给定GO ID的所有父节点（祖先）的ID集合。
    """
    parents = set()
    if include_self:
        parents.add(go_id)
    
    if go_id not in obo_dag:
        return parents # 如果GO ID不在图中，返回空集合或包含自身的集合

    # 使用一个队列进行广度优先或深度优先搜索来遍历父节点
    terms_to_visit = {obo_dag[go_id].id} # 从当前GO Term的ID开始
    visited_terms = set()

    while terms_to_visit:
        current_term = terms_to_visit.pop()
        if current_term in visited_terms:
            continue
        visited_terms.add(current_term)
        
        # 将当前term的直接父节点加入待访问列表，并将它们的ID加入结果集
        for parent_term in obo_dag[current_term].parents:
            parents.add(parent_term.id)
            if parent_term.id not in visited_terms: # 避免重复访问已经处理过的父节点
                 terms_to_visit.add(parent_term.id)
    return parents

def map_go_to_custom_classes(
    original_go_annotations_dict: Dict[str, List[str]],
    target_classes_dict: Dict[str, str],
    obo_dag: GODag,
) -> Dict[str, List[str]]:
    """
    将原始GO注释字典映射到定义的目标大类上。

    参数:
        original_go_annotations_dict (dict): {protein_id: [original_go_id1, original_go_id2, ...]}
        target_classes_dict (dict): {class_name: target_go_id}
        obo_dag (GODag): 已加载的GO本体对象 (来自goatools)

    返回:
        mapped_annotations_dict (dict): {protein_id: [target_class_name1, target_class_name2, ...]}
                                        或者 {protein_id: [bool_vec_for_classes]} 如果你想直接输出二值化
                                        这里返回类别名称列表更灵活，后续再用MultiLabelBinarizer
    """
    mapped_annotations_dict = defaultdict(list)
    target_class_names = list(target_classes_dict.keys()) # 保持顺序
    target_class_go_ids = list(target_classes_dict.values())

    for protein_id, original_go_ids in original_go_annotations_dict.items():
        protein_belongs_to_classes = set() # 用set避免重复添加同一个大类名
        if not original_go_ids: # 如果某个蛋白没有原始GO注释
            continue

        for original_go_id in original_go_ids:
            original_term = obo_dag.get(original_go_id)
            if not original_term: # 如果OBO文件中没有这个原始GO ID（不太可能，但以防万一）
                # print(f"警告: 原始GO ID {original_go_id} 在OBO文件中未找到。")
                continue
            
            # 获取 original_go_id 的所有祖先（包括它自己）
            # 注意：get_all_parents_of_term 返回的是ID字符串的集合
            original_term_ancestors_and_self = get_all_parents_of_term(original_go_id, obo_dag, include_self=True)

            for i, target_go_id in enumerate(target_class_go_ids):
                if target_go_id not in obo_dag:
                    continue
                
                # 判断 target_go_id 是否是 original_go_id 的祖先 (或 original_go_id 本身)
                if target_go_id in original_term_ancestors_and_self:
                    protein_belongs_to_classes.add(target_class_names[i])
        
        if protein_belongs_to_classes:
            mapped_annotations_dict[protein_id] = sorted(list(protein_belongs_to_classes))

    return dict(mapped_annotations_dict)# 标签注释编码


# --- 标签编码函数 ---
def encode_annotations(
    annotations: Dict[str, List[str]]
) -> Tuple[List[str], np.ndarray, MultiLabelBinarizer]:
    """将注释字典编码成 Multi-Label Binarized 矩阵。"""
    # (代码与之前版本相同)
    print("编码标签...")
    if not annotations:
        print("错误: 没有注释数据可供编码。")
        return [], np.array([]), MultiLabelBinarizer()

    ids = list(annotations.keys())
    labels = list(annotations.values())

    mlb = MultiLabelBinarizer()
    try:
        # 显式传递所有可能的类别，确保顺序一致性 (特别是对于 custom mapping)
        all_possible_labels = sorted(list(set(lbl for sublist in labels for lbl in sublist)))
        mlb.fit(labels) # fit 确定所有类别
        # 或者 mlb.fit([all_possible_labels]) # 确保所有类别都被学习
        labels_encoded = mlb.transform(labels)

        print(f"标签编码完成。类别数量: {len(mlb.classes_)}")
        return ids, labels_encoded.astype(np.float32), mlb
    except Exception as e:
        print(f"标签编码时出错: {e}")
        return [], np.array([]), mlb


# --- 标签分布统计函数 ---
def save_label_distribution(
    encoded_labels: np.ndarray,
    mlb: MultiLabelBinarizer,
    output_file: str = "label_distribution.txt"
):
    """统计并保存最终标签的分布情况。"""
    # (代码与之前版本相同)
    print(f"统计标签分布并保存到 {output_file}...")
    try:
        label_counts = encoded_labels.sum(axis=0)
        class_names = mlb.classes_
        total_samples = encoded_labels.shape[0]

        if len(label_counts) != len(class_names):
             print(f"错误：标签计数 ({len(label_counts)}) 与类别名称数量 ({len(class_names)}) 不匹配。")
             return

        distribution_data = []
        for name, count in zip(class_names, label_counts):
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            distribution_data.append({
                "label": name,
                "count": int(count),
                "percentage": percentage
            })

        sorted_distribution = sorted(distribution_data, key=lambda x: x['count'], reverse=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("最终模型训练标签分布统计：\n")
            f.write("=" * 50 + "\n")
            f.write(f"总样本数量: {total_samples}\n")
            f.write(f"总类别数量: {len(class_names)}\n")
            f.write("=" * 50 + "\n\n")
            for item in sorted_distribution:
                f.write(f"类别: {item['label']}\n")
                f.write(f"  蛋白质数量: {item['count']}\n")
                f.write(f"  占比: {item['percentage']:.2f}%\n")
                f.write("-" * 30 + "\n")
        print("标签分布统计完成。")
    except Exception as e:
        print(f"保存标签分布时出错: {e}")


# --- 映射诊断函数 ---
import random
def diagnose_mapping(
    original_annotations: Dict[str, List[str]],
    mapped_annotations: Dict[str, List[str]],
    go_dag: GODag,
    num_samples: int = 5
):
    """随机抽样展示原始 GO 注释和映射后类别的对比。"""
    # (代码与之前版本相同)
    print("\n--- 开始映射诊断 (抽样) ---")
    protein_ids = list(mapped_annotations.keys())
    if len(protein_ids) == 0:
        print("没有可供诊断的映射后注释。")
        return
    if len(protein_ids) < num_samples:
        sample_ids = protein_ids
    else:
        sample_ids = random.sample(protein_ids, num_samples)

    for pid in sample_ids:
        print(f"\n蛋白质 ID: {pid}")
        orig_gos = original_annotations.get(pid, [])
        print("  原始 GO Terms (筛选后):")
        if orig_gos:
            for go_id in orig_gos:
                term = go_dag.get(go_id)
                print(f"    - {go_id} ({term.name if term else 'N/A'})")
        else:
            print("    - (无)")

        mapped_cats = mapped_annotations.get(pid, [])
        print("  映射后类别:")
        if mapped_cats:
            for cat_id_or_name in mapped_cats:
                term = go_dag.get(cat_id_or_name) # Slim ID 也是 GO ID
                if term:
                     print(f"    - {cat_id_or_name} ({term.name})")
                else: # 自定义类别名称
                     print(f"    - {cat_id_or_name}")
        else:
            print("    - (无)")
    print("--- 映射诊断结束 ---")