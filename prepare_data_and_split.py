# prepare_data_and_split.py
import argparse
import os
import sys
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from goatools.obo_parser import GODag

# --- 导入自定义模块 ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import config
from data_utils import (
    parse_uniprot_dat, load_data_from_excel, filter_annotations_by_category,
    build_go_to_slim_map, fast_map_to_slim, map_go_to_custom_classes,
    # encode_annotations is effectively replaced by direct mlb usage here
    save_label_distribution # 可以用这个来查看划分后的标签分布
)
from feature_extractor import extract_protbert_features_batch

def main_prepare_split(args):
    print("--- 开始数据准备与固定划分 ---")
    os.makedirs(args.output_dir, exist_ok=True) # 创建输出目录

    # --- 1. 加载 GO 本体 ---
    print("\n--- 加载 GO 本体文件 ---")
    go_dag = GODag(config.OBO_FILE, optional_attrs={'relationship'})
    slim_terms = None
    if args.mapping_strategy == 'goslim':
        try:
            slim_dag = GODag(config.SLIM_OBO_FILE)
            slim_terms = set(slim_dag.keys())
        except FileNotFoundError:
            print(f"错误: GO Slim OBO 文件 '{config.SLIM_OBO_FILE}' 未找到。")
            return

    # --- 2. 加载原始数据 ---
    print(f"\n--- 加载输入数据 ({args.input_data_file}) ---")
    # ... (与 main.py 中类似的加载逻辑) ...
    file_ext = os.path.splitext(args.input_data_file)[1].lower()
    if file_ext == '.dat':
        sequences, go_annotations, go_categories = parse_uniprot_dat(args.input_data_file)
    elif file_ext in ['.xlsx', '.xls']:
        sequences, go_annotations, go_categories = load_data_from_excel(args.input_data_file, go_col=args.go_col_name)
    else:
        print(f"错误: 不支持的文件格式 '{file_ext}'。")
        return
    if not sequences: return

    # --- 3. 标签预处理 (类别筛选 + 映射) ---
    print(f"\n--- 按类别 '{args.target_go_category}' 筛选并进行 '{args.mapping_strategy}' 映射 ---")
    # ... (与 main.py 中类似的筛选和映射逻辑) ...
    all_prot_ids = set(sequences.keys())
    for prot_id in all_prot_ids:
        if prot_id not in go_categories: go_categories[prot_id] = {'MF': [], 'BP': [], 'CC': []}
        if prot_id not in go_annotations and prot_id in go_categories:
            all_ids_in_cats = set(go for cat_list in go_categories[prot_id].values() for go in cat_list)
            if all_ids_in_cats: go_annotations[prot_id] = list(all_ids_in_cats)

    category_annotations = filter_annotations_by_category(go_annotations, go_categories, args.target_go_category, go_dag)
    if not category_annotations: print("错误: 类别筛选后无数据"); return

    if args.mapping_strategy == 'goslim':
        # 考虑也缓存 go_to_slim_map
        cache_slim_map_file = os.path.join(args.output_dir, "go_to_slim_map.pkl")
        if os.path.exists(cache_slim_map_file):
            with open(cache_slim_map_file, 'rb') as f: go_to_slim_map = pickle.load(f)
            print("已加载缓存的GO Slim映射表。")
        else:
            go_to_slim_map = build_go_to_slim_map(go_dag, slim_terms)
            with open(cache_slim_map_file, 'wb') as f: pickle.dump(go_to_slim_map, f)
            print("GO Slim映射表已构建并缓存。")
        final_mapped_annotations = fast_map_to_slim(category_annotations, go_to_slim_map)
    elif args.mapping_strategy == 'custom':
        final_mapped_annotations = map_go_to_custom_classes(category_annotations, config.TARGET_MF_CLASSES, go_dag, args.target_go_category)
    else: print("错误：未知的映射策略"); return

    if not final_mapped_annotations: print("错误: 标签映射后无数据"); return

    # --- 4. 准备用于 MultiLabelBinarizer 的数据 ---
    all_pids_after_mapping = []
    all_labels_original_format = []
    for pid, mapped_gos in final_mapped_annotations.items():
        if pid in sequences: # 确保序列存在
            all_pids_after_mapping.append(pid)
            all_labels_original_format.append(mapped_gos)

    if not all_pids_after_mapping: print("错误: 映射后无有效蛋白质ID"); return

    # --- 5. 创建并拟合 MultiLabelBinarizer，然后编码标签 ---
    mlb = MultiLabelBinarizer()
    # 在所有可用数据上拟合 mlb，以确保它学习到所有可能的标签
    all_encoded_labels = mlb.fit_transform(all_labels_original_format).astype(np.float32)
    print(f"MultiLabelBinarizer 拟合完成。类别数: {len(mlb.classes_)}")
    print(f"MLB 类别: {mlb.classes_}")

    # 保存 mlb 对象
    mlb_save_path = os.path.join(args.output_dir, "mlb_encoder.pkl")
    with open(mlb_save_path, 'wb') as f:
        pickle.dump(mlb, f)
    print(f"MLB 编码器已保存到: {mlb_save_path}")

    # --- 6. 提取所有有效蛋白质的特征 ---
    print("\n--- 提取 ProtBERT 特征 ---")
    all_sequences_to_extract = {pid: sequences[pid] for pid in all_pids_after_mapping}
    all_features_dict = extract_protbert_features_batch(
        all_sequences_to_extract, config.CACHE_DIR, args.batch_size_feature_extraction,
        config.MAX_SEQ_LENGTH, config.DEVICE
    )

    # 对齐特征、标签和ID (基于 all_pids_after_mapping 的顺序)
    final_pids = []
    final_features_list = []
    final_encoded_labels_list = [] # 将用于切分

    for i, pid in enumerate(all_pids_after_mapping):
        if pid in all_features_dict:
            final_pids.append(pid)
            final_features_list.append(all_features_dict[pid])
            final_encoded_labels_list.append(all_encoded_labels[i]) # all_encoded_labels 是按 all_pids_after_mapping 顺序的

    if not final_pids: print("错误: 特征提取或对齐后无数据"); return
    final_encoded_labels_array = np.array(final_encoded_labels_list) # 转换为Numpy数组

    print(f"特征提取和对齐完成。最终样本数: {len(final_pids)}")

    # --- 7. 执行固定的数据划分 (训练集、验证集、测试集) ---
    print("\n--- 执行数据划分 (使用固定随机种子) ---")
    # 准备分层标签 (与 main.py 中逻辑类似，但作用于完整数据集)
    stratify_source = None
    if len(final_pids) > 1 and final_encoded_labels_array.shape[1] > 0:
        try:
            label_tuples = [tuple(row) for row in final_encoded_labels_array]
            # ... (可以加入更复杂的分层逻辑，如果需要)
            stratify_source = label_tuples
        except: stratify_source = final_encoded_labels_array.argmax(axis=1) # 简化

    # 划分测试集
    train_val_indices, test_indices = train_test_split(
        np.arange(len(final_pids)),
        test_size=args.test_set_ratio,
        random_state=args.random_seed, # 固定随机种子
        stratify=stratify_source
    )
    # 从剩余的划分验证集
    stratify_train_val = None
    if stratify_source is not None:
        stratify_train_val = [stratify_source[i] for i in train_val_indices]

    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=args.validation_set_ratio / (1 - args.test_set_ratio), # 在非测试集中划分验证集
        random_state=args.random_seed, # 固定随机种子
        stratify=stratify_train_val
    )

    # --- 8. 保存划分后的数据 ---
    data_splits = {
        'train': {'ids': [], 'features': [], 'labels': []},
        'validation': {'ids': [], 'features': [], 'labels': []},
        'test': {'ids': [], 'features': [], 'labels': []}
    }

    for i in train_indices:
        data_splits['train']['ids'].append(final_pids[i])
        data_splits['train']['features'].append(final_features_list[i])
        data_splits['train']['labels'].append(final_encoded_labels_array[i])

    for i in val_indices:
        data_splits['validation']['ids'].append(final_pids[i])
        data_splits['validation']['features'].append(final_features_list[i])
        data_splits['validation']['labels'].append(final_encoded_labels_array[i])

    for i in test_indices:
        data_splits['test']['ids'].append(final_pids[i])
        data_splits['test']['features'].append(final_features_list[i])
        data_splits['test']['labels'].append(final_encoded_labels_array[i])

    # 将每个split的数据转换为Numpy数组（如果需要）并保存
    for split_name, split_data in data_splits.items():
        if not split_data['ids']:
            print(f"警告: {split_name} 集为空。")
            continue
        
        # 将特征列表转换为 (N, D) Numpy 数组
        features_array = np.array(split_data['features'])
        labels_array = np.array(split_data['labels'])

        np.save(os.path.join(args.output_dir, f"{split_name}_ids.npy"), np.array(split_data['ids']))
        np.save(os.path.join(args.output_dir, f"{split_name}_features.npy"), features_array)
        np.save(os.path.join(args.output_dir, f"{split_name}_labels.npy"), labels_array)
        print(f"已保存 {split_name} 集: {len(split_data['ids'])} 个样本。特征形状: {features_array.shape}, 标签形状: {labels_array.shape}")

        # (可选) 为该划分保存标签分布
        if labels_array.size > 0 :
             save_label_distribution(labels_array, mlb, output_file=os.path.join(args.output_dir, f"{split_name}_label_distribution.txt"))


    print(f"\n--- 数据准备与固定划分完成。数据已保存到 '{args.output_dir}' ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据预处理、特征提取与固定划分脚本")
    parser.add_argument('--input_data_file', type=str, required=True, help='输入数据文件路径')
    parser.add_argument('--output_dir', type=str, default="prepared_data", help='保存处理后数据的目录')
    parser.add_argument('--target_go_category', type=str, default='MF', choices=['MF', 'BP', 'CC'])
    parser.add_argument('--mapping_strategy', type=str, default='goslim', choices=['goslim', 'custom'])
    parser.add_argument('--go_col_name', type=str, default='Gene Ontology (GO)', help='Excel中GO注释列名')
    parser.add_argument('--test_set_ratio', type=float, default=0.2, help='测试集在总数据中的比例')
    parser.add_argument('--validation_set_ratio', type=float, default=0.1, help='验证集在总数据中的比例 (注意：这是指占原始总数据的比例，实际是从 (1-test_set_ratio) 中划分)')
    parser.add_argument('--random_seed', type=int, default=42, help='用于数据划分的随机种子')
    parser.add_argument('--batch_size_feature_extraction', type=int, default=config.BATCH_SIZE, help='特征提取时的批次大小')


    cli_args = parser.parse_args()
    # 确保验证集比例合理
    if cli_args.test_set_ratio + cli_args.validation_set_ratio >= 1.0:
        print("错误：测试集和验证集比例之和不能大于等于1。")
    else:
        main_prepare_split(cli_args)