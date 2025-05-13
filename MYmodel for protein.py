from transformers import BertModel, BertTokenizer
import numpy as np
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import os
from sklearn.metrics import f1_score #,accuracy_score
import torch.nn as nn
from tqdm import tqdm
from Bio import SeqIO
import pandas as pd
from collections import defaultdict
import hashlib

# -------------------- 数据读取与加载 --------------------
# 读取 fasta 序列
def load_fasta(fasta_file):
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id.split('|')[1]] = str(record.seq)
    return sequences

# 读取注释表 (tsv 格式: ID, function)
def load_annotations(annotation_file):
    df = pd.read_csv(annotation_file, sep='\t')
    annotations = df.groupby('Entry')['GO'].apply(list).to_dict()
    return annotations

def load_data_from_excel(excel_file_path, id_col='Entry', seq_col='Sequence', go_col='Gene Ontology (molecular function)'):
    """
    从Excel文件中加载蛋白质序列和GO注释。

    参数:
        excel_file_path (str): Excel文件的路径。
        id_col (str): 包含蛋白质ID的列名。
        seq_col (str): 包含蛋白质序列的列名。
        go_col (str): 包含GO注释的列名。GO注释在该列中可能是以分号分隔的字符串。

    返回:
        sequences_dict (dict): {protein_id: sequence_string}
        annotations_dict (dict): {protein_id: [go_term1, go_term2, ...]}
    """
    try:
        df = pd.read_excel(excel_file_path)
    except FileNotFoundError:
        print(f"错误: Excel文件未找到于路径 {excel_file_path}")
        return {}, {}
    except Exception as e:
        print(f"读取Excel文件时发生错误: {e}")
        return {}, {}

    sequences_dict = {}
    annotations_dict = defaultdict(list)

    # 检查列是否存在
    required_cols = [id_col, seq_col, go_col]
    for col in required_cols:
        if col not in df.columns:
            print(f"错误: Excel文件中未找到必需的列 '{col}'。可用列: {df.columns.tolist()}")
            return {}, {}

    for index, row in df.iterrows():
        protein_id = str(row[id_col]).strip()
        sequence = str(row[seq_col]).strip().upper() # 转换为大写并去除空格
        go_terms_str = str(row[go_col]) # 转换为字符串以防是数字或其他类型

        if not protein_id or not sequence:
            print(f"警告: 第 {index+2} 行缺少蛋白质ID或序列，已跳过。")
            continue

        # # 确保序列只包含合法的氨基酸字符 (可选，但推荐)
        # valid_aa = "ACDEFGHIKLMNPQRSTVWY"
        # sequence_cleaned = "".join(filter(lambda char: char in valid_aa, sequence))
        # if len(sequence_cleaned) != len(sequence):
        #     print(f"警告: 蛋白质 {protein_id} 的序列包含无效字符，已清理。原始: {sequence}, 清理后: {sequence_cleaned}")
        # sequence = sequence_cleaned
        # if not sequence: # 清理后可能为空
        #     print(f"警告: 蛋白质 {protein_id} 清理后序列为空，已跳过。")
        #     continue
            
        sequences_dict[protein_id] = sequence

        # 处理GO注释，假设它们是以分号分隔的字符串，并且我们只取GO ID部分
        # 例如 "GO:0005575 cellular component; GO:0003677 DNA binding"
        # 或者可能直接是 "structural molecule activity; RNA binding" (这种需要映射到GO ID)
        # 为了简化，我们先假设go_col直接包含GO ID，或者可以提取出GO ID的模式
        
        # 这是一个非常简化的GO提取，实际中你可能需要更复杂的逻辑
        # 如果你的GO列直接是GO ID列表（如 "GO:xxxxxxx;GO:yyyyyyy"）:
        if isinstance(go_terms_str, str) and go_terms_str.strip():
            # 移除GO名称和证据代码，只保留GO ID
            # 例如，从 "GO:0005575 cellular component" 提取 "GO:0005575"
            import re
            # 正则表达式匹配 GO: followed by 7 digits
            found_go_ids = re.findall(r"GO:\d{7}", go_terms_str)
            if found_go_ids:
                annotations_dict[protein_id].extend(found_go_ids)
            else:
                # 如果GO列是功能描述而不是GO ID，这里需要一个映射步骤
                # 例如，"structural molecule activity" -> 对应的GO ID
                # 这通常需要一个预先构建的 名称->ID 的映射表
                print(f"警告: 蛋白质 {protein_id} 的GO注释 '{go_terms_str}' 中未找到标准GO ID格式。此蛋白质的注释可能不完整或需要额外处理。")
                # 作为一个占位符，如果不是GO ID，你可能需要一个转换函数
                # annotations_dict[protein_id].append(convert_description_to_goid(go_terms_str))

    # 清理那些有序列但没有成功解析出任何GO注释的条目（如果需要）
    # final_annotations_dict = {pid: gos for pid, gos in annotations_dict.items() if gos}
    # sequences_dict_filtered = {pid: seq for pid, seq in sequences_dict.items() if pid in final_annotations_dict}
    # return sequences_dict_filtered, final_annotations_dict
       
    return sequences_dict, dict(annotations_dict) # 将defaultdict转为普通dict返回

# 针对.dat格式编写的读取与加载
def parse_uniprot_dat(file_path):
    with open(file_path, 'r') as f:
        data = f.read()

    entries = data.strip().split('//\n')
    sequences = {}
    go_annotations = {}
    go_categories = {}

    for entry in entries:
        if not entry.strip():
            continue

        # 提取 AC
        ac_match = re.search(r'^AC\s+(.+);', entry, re.MULTILINE)
        ac = ac_match.group(1).split(';')[0].strip() if ac_match else ''

        # 提取序列
        seq_match = re.search(r'SQ\s+SEQUENCE.+?\n(.+)', entry, re.DOTALL)
        sequence = ''
        if seq_match:
            seq_block = seq_match.group(1)
            sequence = ''.join(seq_block.split()).replace('\n', '').replace(' ', '')
        sequences[ac] = sequence

        # 提取 GO 注释
        go_matches = re.findall(r'DR\s+GO;\s+(GO:\d+);\s+([CPF]):', entry)
        go_ids = [go_id for go_id, _ in go_matches]
        sequences[ac] = sequence
        go_annotations[ac] = go_ids

        # 分类记录 GO → MF, BP, CC
        go_cats = {'MF': [], 'BP': [], 'CC': []}
        for go_id, cat in go_matches:
            if cat == 'F':
                go_cats['MF'].append(go_id)
            elif cat == 'P':
                go_cats['BP'].append(go_id)
            elif cat == 'C':
                go_cats['CC'].append(go_id)
        go_categories[ac] = go_cats

    return sequences, go_annotations, go_categories

# 你需要先定义好你的7个目标大类及其对应的GO ID
TARGET_MF_CLASSES = {
    "Catalytic_Activity": "GO:0003824",
    "Binding": "GO:0005488", # 或者更细致的子类
    "Transporter_Activity": "GO:0005215",
    "Structural_Molecule_Activity": "GO:0005198",
    "Transcription_Regulation_Activity": "GO:0030528", # 或 GO:0003700
    "Signal_Transduction_Activity": "GO:0060089",     # 或 GO:0038023
    "Chaperone_Folding_Activity": "GO:0061077"      # 或其他代表性的 chaperone GO ID
}
# 确保这些是你最终确定的ID

# 你还需要GO本体文件来检查层级关系
from goatools.obo_parser import GODag
from goatools.gosubdag.gosubdag import GoSubDag
obodag = GODag("go-basic.obo") # 确保这个文件存在

def get_all_parents_of_term(go_id, obo_dag, include_self=False):
    """
    获取给定GO ID的所有父节点（祖先）的ID集合。
    """
    parents = set()
    if include_self:
        parents.add(go_id)
    
    if go_id not in obo_dag:
        return parents # 如果GO ID不在图中，返回空集合或包含自身的集合

    # 使用一个队列进行广度优先或深度优先搜索来遍历父节点
    terms_to_visit = {obo_dag[go_id]} # 从当前GO Term对象开始
    visited_terms = set()

    while terms_to_visit:
        current_term = terms_to_visit.pop()
        if current_term in visited_terms:
            continue
        visited_terms.add(current_term)
        
        # 将当前term的直接父节点加入待访问列表，并将它们的ID加入结果集
        for parent_term in current_term.parents:
            parents.add(parent_term.id)
            if parent_term not in visited_terms: # 避免重复访问已经处理过的父节点
                 terms_to_visit.add(parent_term)
    return parents
def map_go_to_target_classes(original_go_annotations_dict, target_classes_dict=TARGET_MF_CLASSES, obo_dag=obodag):
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

def save_label_distribution(mapped_annotations_dict, output_file="label_distribution.txt"):
    """
    统计并保存每个标签对应的蛋白质数量
    
    参数:
        mapped_annotations_dict: 映射后的注释字典 {protein_id: [class1, class2, ...]}
        output_file: 输出文件名
    """
    # 统计每个标签的蛋白质数量
    label_counts = defaultdict(int)
    for protein_classes in mapped_annotations_dict.values():
        for class_name in protein_classes:
            label_counts[class_name] += 1
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("标签分布统计：\n")
        f.write("=" * 50 + "\n")
        f.write(f"总蛋白质数量: {len(mapped_annotations_dict)}\n")
        f.write("=" * 50 + "\n\n")
        
        # 按数量降序排序
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        
        for label, count in sorted_labels:
            percentage = (count / len(mapped_annotations_dict)) * 100
            f.write(f"{label}:\n")
            f.write(f"  蛋白质数量: {count}\n")
            f.write(f"  占比: {percentage:.2f}%\n")
            f.write("-" * 30 + "\n")
def encode_annotations(annotations):
    """
    将注释字典编码成 one-hot 矩阵。
    
    参数：
        annotations (dict): 
            形如 {"P12345": ["GO:00001", "GO:00002"], ...} 的字典。
    
    返回：
        ids (list): 蛋白质 ID 列表。
        labels_encoded (ndarray): one-hot 编码后的标签数组。
        mlb (MultiLabelBinarizer): 训练好的 MultiLabelBinarizer 对象，用于反解码或新样本编码。
    """
    ids = list(annotations.keys())
    labels = list(annotations.values())
    
    mlb = MultiLabelBinarizer()
    labels_encoded = mlb.fit_transform(labels)
    
    return ids, labels_encoded, mlb

# -------------------- 数据预处理部分 --------------------
# 蛋白质编码
# 加载 ProtBERT
#tokenizer：分词器
#protBERT用法定义
def extract_protbert_features(sequence):
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    tokmodel = BertModel.from_pretrained("Rostlab/prot_bert")
    max_length = 1024     

    """使用 ProtBERT 处理蛋白质序列"""
    sequence = " ".join(sequence)  # 在氨基酸之间添加空格
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = tokmodel(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # 取平均池化
    #[CLS] 向量（位置0的嵌入）
    # return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    # return outputs.last_hidden_state.squeeze(0).numpy() 

def extract_protbert_features_batch(sequences, batch_size=32, cache_dir="features"):
    """批量提取特征，支持缓存机制"""
    # 检查是否有可用的 CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    
    # 初始化结果列表
    all_features = []
    sequences_to_process = []
    sequence_indices = []
    
    # 首先检查缓存
    for idx, seq in enumerate(sequences):
        # 为每个序列生成唯一的缓存文件名
        cache_key = hashlib.md5(seq.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{cache_key}.npy")
        
        if os.path.exists(cache_path):
            # 如果缓存存在，直接加载
            feature = np.load(cache_path)
            all_features.append(feature)
        else:
            # 如果缓存不存在，添加到待处理列表
            sequences_to_process.append(seq)
            sequence_indices.append(idx)
    
    # 如果有需要处理的序列
    if sequences_to_process:
        print(f"需要处理 {len(sequences_to_process)} 个新序列")
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        model = BertModel.from_pretrained("Rostlab/prot_bert").to(device)
        model.eval()
        
        # 处理新序列
        for i in tqdm(range(0, len(sequences_to_process), batch_size), desc="提取ProtBERT特征"):
            batch_sequences = sequences_to_process[i:i + batch_size]
            # 在氨基酸之间添加空格
            batch_sequences = [" ".join(seq) for seq in batch_sequences]
            
            # 批量处理
            inputs = tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                
                # 保存新特征到缓存
                for j, feature in enumerate(features):
                    seq_idx = i + j
                    if seq_idx < len(sequences_to_process):
                        cache_key = hashlib.md5(sequences_to_process[seq_idx].encode()).hexdigest()
                        cache_path = os.path.join(cache_dir, f"{cache_key}.npy")
                        np.save(cache_path, feature)
                
                # 将特征添加到结果列表
                for j, feature in enumerate(features):
                    seq_idx = i + j
                    if seq_idx < len(sequences_to_process):
                        all_features.insert(sequence_indices[seq_idx], feature)
    
    return np.array(all_features)

class ProteinDataset(Dataset):
    """蛋白质数据集类"""
    def __init__(self, features, labels, max_length=1024):
        self.features = features  # 直接使用预计算的特征
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.tensor(
            self.labels[idx],
            dtype=torch.float if isinstance(self.labels[0], (list, np.ndarray)) else torch.long
        )

    @property
    def is_multilabel(self):
        return isinstance(self.labels[0], (list, np.ndarray)) and len(self.labels[0]) > 1



#到此为止protein的原始序列通过ProtBERT被转译成AI可以理解的数学形式（保留特征）

# -------------------- 模型 --------------------

#定义LSTM+Attention模型
class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, multi_label=False):
        super(BiLSTMAttention, self).__init__()
        self.multi_label = multi_label
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.activation = nn.Sigmoid() if multi_label else nn.Softmax(dim=1)
    
    def forward(self, x):
        # 如果输入 x 的形状是 (batch_size, input_dim)，需要增加一个序列长度维度
        if x.ndim == 2:
            x = x.unsqueeze(1) # (batch_size, input_dim) -> (batch_size, 1, input_dim)

        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_dim*2)
        logits = self.fc(context)  # (batch, output_dim)
        return self.activation(logits)  # 根据任务选择激活函数
    
#定义CNN + BiLSTM模型
class CNN_BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=3, num_layers=1, multi_label=False):
        super(CNN_BiLSTM, self).__init__()
        self.multi_label = multi_label
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.activation = nn.Sigmoid() if multi_label else nn.Softmax(dim=1)

    def forward(self, x):
        # 如果输入 x 的形状是 (batch_size, input_dim)，需要增加一个序列长度维度
        if x.ndim == 2:
            x = x.unsqueeze(1) # (batch_size, input_dim) -> (batch_size, 1, input_dim)

        x = x.permute(0, 2, 1)  # (batch, input_dim, seq_len) for CNN
        cnn_out = torch.relu(self.conv1d(x))  # (batch, hidden_dim, seq_len)
        cnn_out = cnn_out.permute(0, 2, 1)  # (batch, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(cnn_out)  # (batch, seq_len, hidden_dim*2)
        logits = self.fc(lstm_out[:, -1, :])  # 取最后一个时间步的输出
        return self.activation(logits)  # 根据任务选择激活函数


# -------------------- 集成模型定义 --------------------
class EnsembleModel(nn.Module):
    """集成模型：加权投票机制"""
    def __init__(self, modelA, modelB, weightA=0.5, multi_label=False):
        super(EnsembleModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.weightA = weightA  # 模型A的权重
        self.multi_label = multi_label

    def forward(self, x):
        outA = self.modelA(x)
        outB = self.modelB(x)
        # 加权平均（软投票）
        combined = self.weightA * outA + (1 - self.weightA) * outB
        if not self.multi_label:  # 多分类需保持概率和为1
            combined = torch.softmax(combined, dim=1)
        return combined



#到上面为止模型定义好了三种（2+1），然后要去输入数据对模型进行训练了，下面如何实现训练过程
# -------------------- 训练准备 --------------------
def prepare_dataloaders(sequences, labels, batch_size=32, test_size=0.2, val_size=0.1, max_length=512):
    """
    准备数据加载器
    返回：train_loader, val_loader, test_loader, num_classes
    """
    num_samples = len(sequences)
    indices = np.arange(num_samples)

    # 先分出测试集
    X_temp_idx, X_test_idx, y_temp, y_test = train_test_split(
        indices, labels,
        test_size=test_size,
        random_state=42
    )

    # 再从剩余数据中分出验证集
    X_train_idx, X_val_idx, y_train, y_val = train_test_split(
        X_temp_idx, y_temp,
        test_size=val_size/(1-test_size),
        random_state=42
    )

    # 预计算所有特征
    print("预计算特征...")
    all_features = extract_protbert_features_batch(sequences)
    
    # 根据索引获取特征和标签
    X_train = [all_features[i] for i in X_train_idx]
    X_val = [all_features[i] for i in X_val_idx]
    X_test = [all_features[i] for i in X_test_idx]

    # 创建数据集
    train_dataset = ProteinDataset(X_train, y_train, max_length)
    val_dataset = ProteinDataset(X_val, y_val, max_length)
    test_dataset = ProteinDataset(X_test, y_test, max_length)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    num_classes = labels.shape[1]
    multilabel = True

    return train_loader, val_loader, test_loader, num_classes, multilabel

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    """
    模型训练函数
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 训练设备
    返回:
        train_losses: 训练损失记录
        val_losses: 验证损失记录
        val_accuracies: 验证准确率记录
    """
    model.to(device)
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # 训练阶段
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device).squeeze()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).squeeze()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total
        
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {epoch_loss:.4f} - '
              f'Val Loss: {val_loss:.4f} - '
              f'Val Acc: {val_accuracy:.4f}')
    
    return train_losses, val_losses, val_accuracies

# -------------------- 集成训练函数 --------------------
def train_ensemble_models(
    modelA, modelB, train_loader, val_loader, 
    criterion, optimizerA, optimizerB, num_epochs=10, device='cpu'
):
    """同时训练两个模型（独立优化器）"""
    models = [modelA.to(device), modelB.to(device)]
    optimizers = [optimizerA, optimizerB]
    
    for epoch in range(num_epochs):
        # 交替训练两个模型
        for model_idx in range(2):
            models[model_idx].train()
            running_loss = 0.0
            
            for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1} Model {model_idx+1}'):
                inputs, labels = inputs.to(device), labels.to(device).squeeze()
                
                optimizers[model_idx].zero_grad()
                outputs = models[model_idx](inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizers[model_idx].step()
                
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Model {model_idx+1} Train Loss: {epoch_loss:.4f}')

# 评估函数
def evaluate_model(model, test_loader, device='cpu'):
    """
    模型评估函数
    参数:
        model: 要评估的模型
        test_loader: 测试数据加载器
        device: 评估设备
    返回:
        accuracy: 准确率
        f1: F1分数
    """
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predicted = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).squeeze()
            outputs = model(inputs)
            
            # 修改这里：确保预测和标签格式一致
            if model.multi_label:
                # 多标签情况：使用阈值0.5
                predicted = (outputs > 0.5).float()
            else:
                # 单标签情况：取最大值
                _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())
    
    accuracy = correct / total
    # 修改这里：根据任务类型选择正确的评估方式
    if model.multi_label:
        f1 = f1_score(all_labels, all_predicted, average='samples')
    else:
        f1 = f1_score(all_labels, all_predicted, average='weighted')
    
    return accuracy, f1


# -------------------- 预测函数 --------------------
def predict_function(sequence, model_path, num_classes,max_length=512):
    """
    蛋白质功能预测函数
    参数：
        sequence: 蛋白质序列字符串
        model_path: 模型权重路径
        max_length: 序列最大长度
    返回：
        preds: 预测结果（概率）
    """
    # 加载模型
    model = BiLSTMAttention(input_dim=1024, hidden_dim=256, output_dim=num_classes)  # 需与训练参数一致
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 特征提取
    features = extract_protbert_features(sequence[:max_length])
    features_tensor = torch.FloatTensor(features).unsqueeze(0)  # 增加batch维度

    # 预测
    with torch.no_grad():
        outputs = model(features_tensor)
    
    return outputs.squeeze().numpy()

# -------------------- 集成预测函数 --------------------
def ensemble_predict(sequence, model_path, num_classes,max_length=512):
    """集成模型预测函数"""
    # 加载所有模型
    checkpoint = torch.load(model_path)
    model_bilstm = BiLSTMAttention(input_dim=1024, hidden_dim=256, output_dim=num_classes)
    model_cnnlstm = CNN_BiLSTM(input_dim=1024, hidden_dim=256, output_dim=num_classes)
    ensemble_model = EnsembleModel(model_bilstm, model_cnnlstm)
    
    model_bilstm.load_state_dict(checkpoint['bilstm'])
    model_cnnlstm.load_state_dict(checkpoint['cnnlstm'])
    ensemble_model.load_state_dict(checkpoint['ensemble'])
    ensemble_model.eval()
    
    # 特征提取
    features = extract_protbert_features(sequence[:max_length])
    features_tensor = torch.FloatTensor(features).unsqueeze(0)
    
    # 预测
    with torch.no_grad():
        outputs = ensemble_model(features_tensor)
    
    return outputs.squeeze().numpy()


# -------------------- 训练配置 --------------------
'''
def main():
    # 示例数据（需替换为真实数据）
    sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
        # ... 更多蛋白质序列
    ]
    labels = [0, 1]  # 单标签示例 或 [[1,0,1], [0,1,0]] 多标签

    # 准备数据加载器
    train_loader, val_loader, test_loader, num_classes, multilabel = prepare_dataloaders(sequences, labels)

    # 模型参数
    input_dim = 1024  # ProtBERT输出维度
    hidden_dim = 256
    output_dim = num_classes

    # 初始化模型（选择其中一个）
    model = BiLSTMAttention(input_dim, hidden_dim, output_dim, multi_label=multilabel)
    # model = CNN_BiLSTM(input_dim, hidden_dim, output_dim, multi_label=multilabel)

    # 损失函数
    criterion = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_losses, val_losses, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=20, device=device
    )

    # 测试
    test_acc, test_f1 = evaluate_model(model, test_loader, device)
    print(f"\nTest Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "protein_function_model.pth")
'''  

# def main1(file_path):
    
#     #读取数据
#     sequences, go_annotations = load_data_from_excel(file_path)
#     cod_annotations  = encode_annotations(go_annotations)
#     protein_data = ProteinDataset(sequences,cod_annotations,extract_protbert_features)
#     features, labels = protein_data[:,:]  #?????
#     train_loader, val_loader, test_loader, num_classes, multilabel = prepare_dataloaders(features, labels)
#     # 模型参数
#     input_dim = 1024  # ProtBERT输出维度
#     hidden_dim = 256  #???
#     output_dim = num_classes
#     criterion = nn.BCEWithLogitsLoss() #损失函数
#     device = 'cpu'
    

    
    
    

#     # 初始化两个独立模型
#     model_bilstm = BiLSTMAttention(input_dim, hidden_dim, output_dim, multi_label=multilabel)
#     model_cnnlstm = CNN_BiLSTM(input_dim, hidden_dim, output_dim, multi_label=multilabel)
    
#     # 定义独立优化器
#     optimizer_bilstm = torch.optim.Adam(model_bilstm.parameters(), lr=1e-4)
#     optimizer_cnnlstm = torch.optim.Adam(model_cnnlstm.parameters(), lr=1e-4)
    
#     # 同时训练两个模型
#     train_ensemble_models(
#         model_bilstm, model_cnnlstm, train_loader, val_loader,
#         criterion, optimizer_bilstm, optimizer_cnnlstm, 
#         num_epochs=20, device = device
#         )
    
#     # 创建集成模型
#     ensemble_model = EnsembleModel(model_bilstm, model_cnnlstm, weightA=0.6, multi_label=multilabel)
    
#     # 评估集成模型
#     test_acc, test_f1 = evaluate_model(ensemble_model, test_loader, device)
#     print(f"\nEnsemble Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")

#     # 保存所有模型
#     torch.save({
#         'bilstm': model_bilstm.state_dict(),
#         'cnnlstm': model_cnnlstm.state_dict(),
#         'ensemble': ensemble_model.state_dict()
#     }, "ensemble_models.pth")

def main_excel(excel_file_path, obo_file_path="go-basic.obo"):

    # 1. 加载OBO本体文件
    print("加载GO本体文件...")
    try:
        obo_dag = GODag(obo_file_path)
    except FileNotFoundError:
        print(f"错误: GO OBO 文件 '{obo_file_path}' 未找到。请确保文件路径正确。")
        return
    except Exception as e:
        print(f"加载OBO文件时出错: {e}")
        return

    # 2. 从Excel读取原始数据
    print(f"从Excel文件加载原始数据: {excel_file_path}...")
    # 假设你的Excel有 'Entry', 'Sequence', 'Gene Ontology (molecular function)' 列
    # GO注释列应该包含原始的、详细的GO ID列表（分号分隔或能解析出来）
    all_sequences_dict, original_go_annotations_dict = load_data_from_excel(
        excel_file_path,
        id_col='Entry',
        seq_col='Sequence',
        go_col='Gene Ontology (molecular function)' # 确保这是你包含原始GO MF ID的列
    )

    if not all_sequences_dict:
        print("未从Excel加载到序列数据。")
        return

    # 3. 将原始GO注释映射到目标大类
    print("将原始GO注释映射到目标大类...")
    # original_go_annotations_dict 应该是 {protein_id: [go_id1, go_id2,...]}
    mapped_annotations_dict = map_go_to_target_classes(
        original_go_annotations_dict,
        TARGET_MF_CLASSES,
        obo_dag
    )
    
    if not mapped_annotations_dict:
        print("没有蛋白质能映射到定义的目标大类。请检查你的大类定义或原始数据。")
        return
    print(f"完成映射，{len(mapped_annotations_dict)} 个蛋白质至少映射到一个大类。")
    # 在main_excel函数中，在映射完成后添加：
    print("将原始GO注释映射到目标大类...")
    mapped_annotations_dict = map_go_to_target_classes(
        original_go_annotations_dict,
        TARGET_MF_CLASSES,
        obo_dag
    )
    # 添加这一行来保存标签分布
    save_label_distribution(mapped_annotations_dict, "label_distribution.txt")

    # 4. 准备用于MultiLabelBinarizer的数据
    # 我们需要一个蛋白质ID列表和对应的映射后的大类名称列表
    protein_ids_for_mlb = []
    mapped_class_names_for_mlb = [] # [[class1, class2], [class3], ...]

    # 确保只处理那些同时存在于序列字典和映射后注释字典中的蛋白质
    valid_protein_ids = set(all_sequences_dict.keys()).intersection(set(mapped_annotations_dict.keys()))
    
    if not valid_protein_ids:
        print("序列数据和映射后的注释数据之间没有共同的蛋白质ID。")
        return

    for pid in valid_protein_ids:
        protein_ids_for_mlb.append(pid)
        mapped_class_names_for_mlb.append(mapped_annotations_dict[pid]) # mapped_annotations_dict[pid] 是一个大类名称列表

    # 5. 对映射后的大类名称进行多标签二值化编码
    print("对映射后的大类进行编码...")
    # encode_annotations 函数需要修改或确认其输入是 {id: [class_name_list]} 或直接是 [class_name_list_for_each_sample]
    # 为了与你现有的 encode_annotations(annotations_dict) 兼容，我们重构一下输入
    temp_dict_for_encode = {pid: mapped_annotations_dict[pid] for pid in protein_ids_for_mlb}
    
    # `encode_annotations` 返回: ids_ordered_by_mlb, labels_encoded_array, mlb_object
    # ids_ordered_by_mlb 是 MultiLabelBinarizer 内部处理过的ID顺序
    # labels_encoded_array 是对应的Numpy二值化标签矩阵
    ids_ordered_by_mlb, final_encoded_labels, mlb = encode_annotations(temp_dict_for_encode)
    # mlb.classes_ 会告诉你编码的顺序，这应该与TARGET_MF_CLASSES的键的顺序一致（如果MultiLabelBinarizer内部排序了）
    # 最好确保MultiLabelBinarizer使用的类别顺序与你期望的一致，可以传递 classes=list(TARGET_MF_CLASSES.keys()) 给它
    
    # 根据mlb处理后的ID顺序，准备最终的序列列表
    final_sequences_list = [all_sequences_dict[pid] for pid in ids_ordered_by_mlb]

    # 6. 准备DataLoaders (使用修改后的版本)
    print("准备DataLoaders...")
    num_classes = final_encoded_labels.shape[1] # 类别数量就是你的大类数量
    multilabel_flag = True # 因为我们是多标签分类

    # 注意：prepare_dataloaders_modified 现在接收的是 序列列表 和 已经编码好的标签Numpy数组
    train_loader, val_loader, test_loader, num_classes, multilabel= prepare_dataloaders(
        final_sequences_list,
        final_encoded_labels,
        batch_size=32, # 举例
        test_size=0.2, # 举例
        val_size=0.1,  # 举例
        max_length=1024 # 或你为ProtBERT设置的长度
    )

    # --- 后续模型训练和评估 ---
    input_dim = 1024
    hidden_dim = 256 # 这是一个超参数，可以调整
    output_dim = num_classes # 等于你的大类数量
    
    # 示例：初始化一个模型
    # model = BiLSTMAttention(input_dim, hidden_dim, output_dim, multi_label=multilabel_flag)
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ... (训练和评估) ...

    print("主函数前半段流程完成。")

    # 初始化两个独立模型
    model_bilstm = BiLSTMAttention(input_dim, hidden_dim, output_dim, multi_label=multilabel)
    model_cnnlstm = CNN_BiLSTM(input_dim, hidden_dim, output_dim, multi_label=multilabel)
    
    # 定义独立优化器
    optimizer_bilstm = torch.optim.Adam(model_bilstm.parameters(), lr=1e-4)
    optimizer_cnnlstm = torch.optim.Adam(model_cnnlstm.parameters(), lr=1e-4)
    
    # 同时训练两个模型
    train_ensemble_models(
        model_bilstm, model_cnnlstm, train_loader, val_loader,
        criterion, optimizer_bilstm, optimizer_cnnlstm, 
        num_epochs=1, device = device
    )
    
    # 创建集成模型
    ensemble_model = EnsembleModel(model_bilstm, model_cnnlstm, weightA=0.6, multi_label=multilabel)
    
    # 评估集成模型
    test_acc, test_f1 = evaluate_model(ensemble_model, test_loader, device)
    print(f"\nEnsemble Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")

    # 保存所有模型
    torch.save({
        'bilstm': model_bilstm.state_dict(),
        'cnnlstm': model_cnnlstm.state_dict(),
        'ensemble': ensemble_model.state_dict()
    }, "ensemble_models.pth")



if __name__ == '__main__':
    # 确保 features 文件夹存在
    if not os.path.exists("features"):
        os.makedirs("features")
    
    uniprot_file = "model\\uniprotkb_AND_reviewed_true_AND_annotat_2025_05_12.xlsx" # <--- 把你的文件名放在这里
    main_excel(uniprot_file)
    
        

        
        
        
        
        
        
        
        