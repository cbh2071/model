# -*- coding: utf-8 -*-
"""
配置文件：存储路径、超参数等
"""
import torch
import os

# --- 文件与目录路径 ---
# 获取当前脚本所在目录的绝对路径
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# !! 用户需要修改以下路径 !!
OBO_FILE = os.path.join(_PROJECT_DIR, "go-basic.obo")           # GO 本体文件路径
SLIM_OBO_FILE = os.path.join(_PROJECT_DIR, "goslim_pir.obo") # GO Slim 文件路径
INPUT_DATA_FILE = os.path.join(_PROJECT_DIR, "dataset_50k.xlsx") # 输入数据文件 (.xlsx 或 .dat)，将在 main.py 通过参数传入
TEST_DATA_FILE = os.path.join(_PROJECT_DIR, "test_data.xlsx") # 测试数据文件 (.xlsx 或 .dat)，将在 evaluate_trained_model.py 通过参数传入
MODEL_PATH = os.path.join(_PROJECT_DIR, "protein_ensemble_model.pth") # 模型保存路径，将在 evaluate_trained_model.py 通过参数传入
GO_COL_NAME = "Gene Ontology (molecular function)" # Excel中GO注释列名，将在 prepare_data_and_split.py 通过参数传入

# --- 自动生成或固定的路径 ---
CACHE_DIR = os.path.join(_PROJECT_DIR, "protbert_features_cache") # ProtBERT 特征缓存目录
MODEL_SAVE_PATH = os.path.join(_PROJECT_DIR, "protein_ensemble_model.pth") # 模型保存路径
DISTRIBUTION_FILE_PREFIX = os.path.join(_PROJECT_DIR, "") # 标签分布文件保存目录（或加前缀）
CHECKPOINT_DIR = os.path.join(_PROJECT_DIR, "checkpoints") # 检查点保存目录
RESULTS_DIR = os.path.join(_PROJECT_DIR, "results") # 结果保存目录
RESUME_CHECKPOINT_BILSTM = os.path.join(_PROJECT_DIR, "checkpoints", "bilstm_model.pth") # BiLSTM 检查点路径
RESUME_CHECKPOINT_CNNLSTM = os.path.join(_PROJECT_DIR, "checkpoints", "cnnlstm_model.pth") # CNN_BiLSTM 检查点路径
PREPARED_DATA_DIR = os.path.join(_PROJECT_DIR, "prepared_data") # 预处理和划分好的数据目录
OUTPUT_DIR = os.path.join(_PROJECT_DIR, "output") # 输出目录

# --- ProtBERT 模型 ---
PROTBERT_MODEL_NAME = "Rostlab/prot_bert"

# --- 数据处理参数 ---
TARGET_GO_CATEGORY = 'MF' # 目标GO类别: 'MF', 'BP', or 'CC' (将在 main.py 通过参数传入)
MAPPING_STRATEGY = 'custom' # 映射策略: 'goslim' 或 'custom' (将在 main.py 通过参数传入)
MIN_SAMPLES_PER_LABEL = 2   # 训练分层划分所需的最小样本数 (目前未使用，分层划分逻辑较复杂)
MAX_SEQ_LENGTH = 1024       # ProtBERT 处理的最大序列长度

# 自定义类别定义 (如果选择 custom 策略)
# !! 用户需要确认这些定义 !!
TARGET_MF_CLASSES = {
    "Catalytic_Activity": "GO:0003824",
    "Binding": "GO:0005488",
    "Transporter_Activity": "GO:0005215",
    "Structural_Molecule_Activity": "GO:0005198",
    "Transcription_Regulation_Activity": "GO:0140110",
    "Molecular_Transducer_Activity": "GO:0060089",
    "Chaperone_Activity": "GO:0003754"
}

# --- 训练参数 ---
BATCH_SIZE = 16             # 批次大小 (可根据显存调整)
NUM_EPOCHS = 200            # 训练轮数
LEARNING_RATE = 1e-4        # 学习率
TEST_SIZE = 0.2             # 测试集比例
VAL_SIZE = 0.1              # 验证集比例 (在训练集中划分)
ENSEMBLE_WEIGHT_A = 0.6     # 集成模型中 BiLSTM 的权重 (模型A)
EARLY_STOPPING_PATIENCE = 5 # 早停策略的耐心值
EARLY_STOPPING_MIN_DELTA = 0.005 # 早停策略的最小变化值
ALPHA = 0.5                 # 类别权重缩放因子
LR_SCHEDULER = 'reducelronplateau' # 学习率调度器
LR_STEP_SIZE = 10           # StepLR 的 step_size
LR_GAMMA = 0.1              # StepLR/ExponentialLR 的 gamma
LR_PATIENCE = 3             # ReduceLROnPlateau 的 patience
LR_FACTOR = 0.1             # ReduceLROnPlateau 的 factor

# --- 模型超参数 ---
INPUT_DIM = 1024            # ProtBERT 输出特征维度
HIDDEN_DIM = 512            # LSTM/CNN 隐藏层维度
NUM_LSTM_LAYERS = 2         # BiLSTM 层数
CNN_KERNEL_SIZE = 3         # CNN 卷积核大小

# --- 设备设置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")