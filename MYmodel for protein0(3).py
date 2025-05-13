from transformers import BertModel, BertTokenizer
import numpy as np
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MultiLabelBinarizer
from sklearn.metrics import f1_score ,accuracy_score
import torch.nn as nn
from tqdm import tqdm
from Bio import SeqIO
import pandas as pd
from goatools.obo_parser import GODag
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import torch.nn.functional as F
import os 
import pickle



tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
tokmodel = BertModel.from_pretrained("Rostlab/prot_bert")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokmodel = tokmodel.to(device)

# -------------------- 数据读取 --------------------
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

# 针对.dat格式编写的读取
def parse_uniprot_dat(file_path):
    with open(file_path, 'r') as f:
        data = f.read()

    entries = re.split(r'//\r?\n', data.strip())
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
        
        go_matches = re.findall(r'^DR\s+GO;\s+(GO:\d+);\s+([CPF]):', entry, re.MULTILINE)

        if go_matches:
           go_ids = [go_id for go_id, _ in go_matches]
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
        
        # 只保留有 GO 注释的条目
    filtered_sequences = {ac: seq for ac, seq in sequences.items() if ac in go_annotations}
    filtered_go_annotations = {ac: go for ac, go in go_annotations.items() if ac in sequences}
    filtered_go_categories = {ac: go_cat for ac, go_cat in go_categories.items() if ac in sequences}

    return filtered_sequences, filtered_go_annotations, filtered_go_categories

#这一步完了之后只留下有go标签的蛋白质序列了

#---------对标签进行处理------------
#go标签很多，所以我们要把go标签通过映射降维
def build_go_to_slim_map(go_dag, slim_terms):
    """预构建所有GO term到其对应slim term的映射字典"""
    go_to_slim = {}
    for go_id in go_dag:
        if go_id in slim_terms:
            go_to_slim[go_id] = {go_id}  # 自身就是slim term
        else:
            # 查找所有祖先中的slim terms
            node = go_dag[go_id]
            slim_parents = {p for p in node.get_all_parents() if p in slim_terms}
            go_to_slim[go_id] = slim_parents
    return go_to_slim

def load_or_create_go_slim_map():
    # 检查缓存文件是否存在
    cache_file = "go_to_slim_map.pkl"
    if os.path.exists(cache_file):
        # 如果缓存文件存在，加载缓存的映射
        with open(cache_file, 'rb') as f:
            goto_slim_map = pickle.load(f)
        print("加载缓存的GO Slim映射")
    else:
        # 如果缓存文件不存在，构建新的映射
        go_dag = GODag("go-basic.obo")
        slim_terms = set(GODag("goslim_generic.obo").keys())
        goto_slim_map = build_go_to_slim_map(go_dag, slim_terms)
        
        # 保存映射到缓存文件
        with open(cache_file, 'wb') as f:
            pickle.dump(goto_slim_map, f)
        print("构建并保存GO Slim映射到缓存")
    
    return goto_slim_map

def fast_map_to_slim(annotation_dict, go_to_slim_map):
    """利用预构建的映射字典快速转换"""
    new_annotations = {}
    for prot_id, go_ids in annotation_dict.items():
        mapped_terms = set()
        for go_id in go_ids:
            if go_id in go_to_slim_map:
                mapped_terms.update(go_to_slim_map[go_id])
        if mapped_terms:
            new_annotations[prot_id] = list(mapped_terms)
    return new_annotations
#这个函数完了以后返回一个字典{蛋白1id:[go_id],蛋白2id:[go_id]}
'''
# 标签注释编码
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
'''
#通过go类型提取专门的go
def collect_go_by_category(go_categories, category='CC'):
    category_go = {}
    for protein_id, cats in go_categories.items():
        if category in cats:
            category_go[protein_id] = cats[category]
    return category_go

# -------------------- 批次处理蛋白质序列，提取蛋白质序列特征 --------------------
def extract_protbert_features_batch(sequences, device="cpu"):
    """批量提取 ProtBERT 特征 """
    #传到这一步的sequence已经是列表了，格式为["abcd","bcde",...,]，长度为batch_size
    
    inputs = tokenizer(
    sequences,                      # 直接传入原始序列（无需手动加空格）
    return_tensors="pt",            # 返回PyTorch张量
    padding=True,                   # 自动填充到批次内最大长度
    truncation=True,                # 自动截断到max_length
    max_length=512,                 # 限制最大长度（包括特殊Token）
    add_special_tokens=True,        # 添加[CLS]和[SEP]（默认True）
    return_attention_mask=True      # 生成attention_mask（默认True）
).to(device)
        
    with torch.no_grad():
        outputs = tokmodel(**inputs)
    
    # 去除 [CLS] 和 [SEP]（ProtBERT tokenizer 的特殊 token）
    attention_mask = inputs['attention_mask']
    hidden_states = outputs.last_hidden_state  # [batch, seq_len, 1024]


    features = []
    for i in range(hidden_states.size(0)): #i表示当前处理的序列在批次中的索引，hidden_states.size(0) = batch_size
        length = attention_mask[i].sum().item()  # 实际总长度（含 [CLS] 和 [SEP]）
        # 截取有效部分：去除 [CLS] (位置0) 和 [SEP] (位置 length-1)
        seq_features = hidden_states[i, 1:length-1, :]  # 形状 [seq_len_i, 1024]
        features.append(seq_features.cpu())
    
    return features  # 返回列表，每个元素形状为 [seq_len_i, 1024]的张量
def collate_fn(batch):
    """ 现在接受原始序列（而非特征） """
    sequences, labels = zip(*batch)
    
    # 在此处调用批量特征提取函数
    features = extract_protbert_features_batch(sequences)  # 返回变长特征列表
    
    # 后续填充逻辑保持不变
    lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    
    # 生成注意力掩码
    attention_mask = torch.arange(padded_features.size(1))[None, :] < lengths[:, None]
    attention_mask = attention_mask.float()
    
    # 处理标签（适配多标签）
    labels = torch.tensor(labels, dtype=torch.float) if isinstance(labels[0], (list, np.ndarray)) else \
             torch.stack(labels)
    
    return padded_features, attention_mask, lengths, labels
# -------------------- 初始原序列+编码好的标签数据存储 --------------------
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels):
        """
        只存储原始序列和标签，特征提取完全交给collate_fn
        """
        self.sequences = sequences  # ["MKYY", "MALW", ...]
        self.labels = labels        # [[0,1], [1,0], ...] 或 [0, 1, ...]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # 直接返回原始序列和标签（不提取特征！）
        return self.sequences[idx], self.labels[idx]

# -------------------- 模型 --------------------

#定义LSTM+Attention模型
class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, multi_label=False):
        super().__init__()
        self.multi_label = multi_label
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        self.attention_layer = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x, attention_mask=None, lengths=None):
        lstm_out, _ = self.lstm(x)
        attention_scores = self.attention_layer(lstm_out).squeeze(-1)
        
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(~attention_mask.bool(), -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=1)
        context = torch.sum(attention_weights.unsqueeze(-1) * lstm_out, dim=1)
        logits = self.fc(context)
        
        # 根据任务类型选择激活函数
        if self.multi_label:
            return torch.sigmoid(logits)
        else:
            return F.softmax(logits, dim=1)
        
class CNN_BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, kernel_size=3, multi_label=False):
        super().__init__()
        self.multi_label = multi_label
        self.conv1d = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        cnn_out = F.relu(self.conv1d(x))
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)
        logits = self.fc(lstm_out[:, -1, :])
        
        if self.multi_label:
            return torch.sigmoid(logits)
        else:
            return F.softmax(logits, dim=1)


# -------------------- 集成模型定义 --------------------
class EnsembleModel(nn.Module):
    """集成模型：加权投票机制"""
    def __init__(self, modelA, modelB, weightA=0.5, multi_label=False):
        super(EnsembleModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.weightA = weightA  # 模型A的权重
        self.multi_label = multi_label

    def forward(self, x, attention_mask=None, lengths=None):
        # 根据子模型类型动态传递参数
        if isinstance(self.modelA, BiLSTMAttention):
            outA = self.modelA(x, attention_mask=attention_mask, lengths=lengths)
        else:
            outA = self.modelA(x)
            
        if isinstance(self.modelB, BiLSTMAttention):
            outB = self.modelB(x, attention_mask=attention_mask, lengths=lengths)
        else:
            outB = self.modelB(x)
            
        combined = self.weightA * outA + (1 - self.weightA) * outB
        return combined



#到上面为止模型定义好了三种（2+1），然后要去输入数据对模型进行训练了，下面如何实现训练过程
# -------------------- 训练准备 --------------------


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            padded_features, attention_mask, lengths, labels = batch
            padded_features = padded_features.to(device)
            labels = labels.float().to(device)  # ⭐ 多标签任务：float 类型

            optimizer.zero_grad()
            
            if isinstance(model, BiLSTMAttention):
                attention_mask = attention_mask.to(device)
                outputs = model(padded_features, attention_mask=attention_mask, lengths=lengths)
            else:
                outputs = model(padded_features)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}')
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                padded_features, attention_mask, lengths, labels = batch
                padded_features = padded_features.to(device)
                labels = labels.float().to(device)  # ⭐ 多标签任务：float 类型

                if isinstance(model, BiLSTMAttention):
                    attention_mask = attention_mask.to(device)
                    outputs = model(padded_features, attention_mask=attention_mask, lengths=lengths)
                else:
                    outputs = model(padded_features)

                val_loss += criterion(outputs, labels).item()
        
        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}')
        
        model.train()


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
def evaluate_model(model, test_loader, device='cpu', is_multilabel=True):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            features, mask, lengths, labels = batch
            features, labels = features.to(device), labels.to(device)
            mask = mask.to(device)

            # Forward
            if isinstance(model, BiLSTMAttention):
                outputs = model(features, attention_mask=mask, lengths=lengths)
            elif isinstance(model, EnsembleModel):
                outputs = model(features, attention_mask=mask, lengths=lengths)
            else:
                outputs = model(features)

            if is_multilabel:
                probs = torch.sigmoid(outputs)           # → 概率
                preds = (probs > 0.5).int()               # → 二值化
            else:
                preds = torch.argmax(outputs, dim=1)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
    
    # 拼接所有批次
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    
    # 评估
    if is_multilabel:
        f1 = f1_score(all_labels, all_preds, average='micro')
        acc = accuracy_score(all_labels, all_preds)  # 可选，不一定适合多标签任务
    else:
        f1 = f1_score(all_labels, all_preds, average='weighted')
        acc = accuracy_score(all_labels, all_preds)

    return acc, f1

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
    features = extract_protbert_features_batch(sequence[:max_length])
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
    features = extract_protbert_features_batch(sequence[:max_length])
    features_tensor = torch.FloatTensor(features).unsqueeze(0)
    
    # 预测
    with torch.no_grad():
        outputs = ensemble_model(features_tensor)
    
    return outputs.squeeze().numpy()

# -------------------- 数据加载 --------------------
def prepare_dataloaders(sequences, labels, batch_size=32, test_size=0.2, val_size=0.1, max_length=512):
     """
     准备训练集、验证集、测试集的DataLoader
     返回：
         train_loader, val_loader, test_loader
     """
     # 划分数据集
     X_train, X_test, y_train, y_test = train_test_split(
         sequences, labels, test_size=test_size, stratify=labels
     )
     X_train, X_val, y_train, y_val = train_test_split(
         X_train, y_train, test_size=val_size, stratify=y_train
     )

     # 创建Dataset
     train_dataset = ProteinDataset(X_train, y_train)
     val_dataset = ProteinDataset(X_val, y_val)
     test_dataset = ProteinDataset(X_test, y_test)

     # 创建DataLoader
     loaders = []
     for dataset in [train_dataset, val_dataset, test_dataset]:
         loader = DataLoader(
             dataset,
             batch_size=batch_size,
             shuffle=(dataset == train_dataset),  # 仅训练集shuffle
             collate_fn=collate_fn,  # 使用你的collate_fn
             pin_memory=True  # 加速GPU传输
         )
         loaders.append(loader)
     # 计算 num_classes 和 is_multilabel
     num_classes = labels.shape[1] if len(labels.shape) > 1 else len(np.unique(labels))
     is_multilabel = len(labels.shape) > 1 and np.any(labels.sum(axis=1) > 1)

     return (*loaders, num_classes, is_multilabel)  # 返回 train_loader, val_loader, test_loader, num_classes, is_multilabel

     

def main1(file_path, n):
    # 1. 数据加载与预处理
    sequences, go_annotations, go_categories = parse_uniprot_dat(file_path)
    
    # 2. 提取MF类别注释并过滤无效数据
    mf_annotations = collect_go_by_category(go_categories, category='MF')
    mf_valid_ids = [k for k in sequences.keys() if k in mf_annotations and mf_annotations[k]]
    print(f"初始有效MF注释蛋白数量: {len(mf_valid_ids)}")

    # 3. GO Slim映射
    goto_slim_map = load_or_create_go_slim_map()
    
    # 应用映射并过滤
    filtered_mf_annotations = {
        k: fast_map_to_slim({k: mf_annotations[k]}, goto_slim_map).get(k, [])
        for k in mf_valid_ids
    }
    filtered_mf_annotations = {k: v for k, v in filtered_mf_annotations.items() if v}
    print(f"映射后有效蛋白数量: {len(filtered_mf_annotations)}")

    # 4. 样本选择
    selected_ids = list(filtered_mf_annotations.keys())[:n]
    final_sequences = [sequences[k] for k in selected_ids]
    final_annotations = [filtered_mf_annotations[k] for k in selected_ids]

    # 5. 标签编码
    mlb = MultiLabelBinarizer()
    encoded_labels = mlb.fit_transform(final_annotations)
    print(f"最终数据集: {len(final_sequences)}个样本, {encoded_labels.shape[1]}个类别")

    # 6. 过滤单样本类别
    label_counts = Counter(map(tuple, encoded_labels))
    valid_indices = [
        i for i, label in enumerate(encoded_labels)
        if label_counts[tuple(label)] >= 2
    ]
    if not valid_indices:
        raise ValueError("所有类别都不足两个样本，无法进行分层划分")
    
    final_sequences = [final_sequences[i] for i in valid_indices]
    encoded_labels = encoded_labels[valid_indices]

    # 7. 准备数据加载器
    train_loader, val_loader, test_loader, num_classes, is_multilabel = \
        prepare_dataloaders(
            sequences=final_sequences,
            labels=encoded_labels,
            batch_size=32,
            test_size=0.2,
            val_size=0.1,
            max_length=512
        )

    # 8. 模型配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {
        "bilstm": BiLSTMAttention(1024, 256, num_classes, multi_label=is_multilabel),
        "cnnlstm": CNN_BiLSTM(1024, 256, num_classes, multi_label=is_multilabel)
    }
    
    # 9. 训练流程
    for name, model in models.items():
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        train_model(
            model, train_loader, val_loader, 
            nn.BCEWithLogitsLoss() if is_multilabel else nn.CrossEntropyLoss(),
            optimizer, num_epochs=50, device=device
        )

    # 10. 集成与评估
    ensemble = EnsembleModel(models["bilstm"], models["cnnlstm"], weightA=0.6)
    test_acc, test_f1 = evaluate_model(ensemble, test_loader, device)
    print(f"集成模型测试结果 - 准确率: {test_acc:.4f}, F1: {test_f1:.4f}")

    # 11. 保存模型和编码器
    torch.save({
        "models": {k: v.state_dict() for k, v in models.items()},
        "ensemble": ensemble.state_dict(),
        "label_encoder": mlb
    }, "protein_function_model.pth")


main1('/Users/mac/Desktop/学习相关/作业文档/医药人工智能/Project/uniprot_sprot.dat',1000)    
    
#max_lenth待会统一一下位置、长度       

        
        
        
        
        
        
        
        