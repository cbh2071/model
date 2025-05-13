# 蛋白质功能预测 (Protein Function Prediction)

本项目旨在使用深度学习模型对蛋白质的分子功能 (Molecular Function, MF) 进行预测。项目整合了多种数据处理技术和模型架构，包括：

*   **数据加载与预处理：** 支持 UniProt DAT 文件和 Excel 文件作为输入。
*   **GO 本体利用：** 使用 Gene Ontology (GO) 和 GO Slim 进行标准化的功能标签映射。同时支持用户自定义的功能大类映射。
*   **特征工程：** 利用预训练的蛋白质语言模型 ProtBERT (from Rostlab) 提取蛋白质序列特征，并采用缓存机制加速重复运行。
*   **模型架构：**
    *   双向长短期记忆网络 (BiLSTM) + Attention机制
    *   一维卷积神经网络 (CNN) + BiLSTM
    *   上述两种模型的加权集成 (Ensemble Model)
*   **训练与评估：** 包含完整的模型训练、验证和多标签评估流程。

## 目录结构

```
protein_function_prediction/
├── config.py                     # 配置文件 (路径, 超参数等)
├── data_utils.py                 # 数据加载, GO映射, 标签编码, 分布统计, 诊断
├── feature_extractor.py          # ProtBERT 特征提取与缓存
├── datasets.py                   # PyTorch Dataset 类
├── models.py                     # 模型定义 (BiLSTM, CNN_LSTM, Ensemble)
├── training_utils.py             # 训练和评估函数
├── main.py                       # 主执行脚本, 参数解析, 流程编排
├── requirements.txt              # Python 依赖库列表
├── go-basic.obo                  # (需要用户提供) GO 本体文件
├── goslim_generic.obo            # (需要用户提供) 通用 GO Slim 文件
├── your_data.xlsx / your_data.dat # (需要用户提供) 输入的蛋白质数据
├── protbert_features_cache/      # (自动创建) ProtBERT 特征缓存目录
├── *_label_distribution.txt      # (自动生成) 标签分布统计文件
└── protein_ensemble_model.pth    # (自动生成) 训练好的模型权重
└── README.md                     # 本文件
```

## 环境要求与安装

1.  **Python 版本：** 建议使用 Python 3.7 或更高版本。
2.  **依赖库安装：**
    首先，克隆或下载本项目。然后，在项目根目录下，通过 pip 安装所需的 Python 包：
    ```bash
    pip install -r requirements.txt
    ```
    主要依赖包括：`torch`, `transformers`, `numpy`, `pandas`, `scikit-learn`, `biopython`, `goatools`, `tqdm`, `openpyxl`。
    **注意：** `torch` 的安装可能需要根据你的 CUDA 版本进行调整。如果你的机器支持 GPU 且已安装 CUDA 和 cuDNN，请确保安装与 CUDA 版本兼容的 PyTorch 版本。

3.  **数据文件准备：**
    *   **GO 本体文件：**
        *   下载 `go-basic.obo` (基础 GO 本体文件) 和 `goslim_generic.obo` (通用 GO Slim 文件) 到项目根目录。这些文件可以从 [Gene Ontology Consortium 官网](http://geneontology.org/docs/download-ontology/) 下载。
        *   如果文件路径不同，请在 `config.py` 中更新 `OBO_FILE` 和 `SLIM_OBO_FILE` 的路径。
    *   **输入蛋白质数据：**
        *   将你的蛋白质数据文件（`.xlsx` 或 `.dat` 格式）放置在项目根目录中，或在运行时通过 `--input_data_file` 参数指定路径。
        *   **Excel (.xlsx) 格式要求：**
            *   至少包含三列：蛋白质 ID (默认为 'Entry')、蛋白质序列 (默认为 'Sequence')、GO 注释 (默认为 'Gene Ontology (GO)'，应包含 `GO:XXXXXXX` 格式的 ID)。
            *   可以在 `data_utils.py` 中的 `load_data_from_excel` 函数调整列名。
        *   **.dat 格式要求：**
            *   标准的 UniProtKB/Swiss-Prot 文本格式。

## 配置

项目的主要配置参数位于 `config.py` 文件中。你可以根据需要修改：

*   **文件路径：** `OBO_FILE`, `SLIM_OBO_FILE` 等。
*   **模型超参数：** `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, `HIDDEN_DIM` 等。
*   **自定义类别：** 如果使用 `--mapping_strategy custom`，请确保 `TARGET_MF_CLASSES` 中的 GO ID 和类别名称符合你的定义。
*   **设备：** `DEVICE` 会自动检测 CUDA 是否可用，并选择 GPU 或 CPU。

## 使用方法

通过命令行运行 `main.py` 脚本来启动整个流程。

**基本命令格式：**

```bash
python main.py --input_data_file <你的数据文件路径> [其他可选参数]
```

**主要命令行参数：**

*   `--input_data_file` (必需): 输入的蛋白质数据文件路径 (例如, `your_data.xlsx` 或 `uniprot_sprot.dat`)。
*   `--target_go_category` (可选): 要预测的 GO 类别。可选值: 'MF' (分子功能), 'BP' (生物过程), 'CC' (细胞组分)。默认为 'MF'。
*   `--mapping_strategy` (可选): 标签映射策略。可选值:
    *   `goslim` (默认): 使用 `goslim_generic.obo` 进行映射。
    *   `custom`: 使用 `config.py` 中定义的 `TARGET_MF_CLASSES` 进行映射。
*   `--batch_size` (可选): 训练和评估的批次大小。默认为 `config.py` 中的设置。
*   `--num_epochs` (可选): 训练轮数。默认为 `config.py` 中的设置。
*   `--learning_rate` (可选): 优化器的学习率。默认为 `config.py` 中的设置。
*   `--ensemble_weight_a` (可选): 集成模型中模型 A (BiLSTM+Attention) 的权重。默认为 `config.py` 中的设置。
*   `--diagnose` (可选): 执行映射诊断步骤，随机抽样展示原始 GO 注释和映射后类别的对比。

**示例命令：**

1.  **使用 GO Slim 映射策略，预测分子功能 (MF)，使用 `my_proteins.xlsx` 作为输入：**
    ```bash
    python main.py --input_data_file my_proteins.xlsx --target_go_category MF --mapping_strategy goslim
    ```

2.  **使用自定义的 7 大类映射策略，预测分子功能 (MF)，使用 `uniprot_data.dat` 作为输入，并进行诊断输出：**
    ```bash
    python main.py --input_data_file uniprot_data.dat --target_go_category MF --mapping_strategy custom --diagnose
    ```

3.  **调整批次大小为 8，训练轮数为 20：**
    ```bash
    python main.py --input_data_file my_proteins.xlsx --batch_size 8 --num_epochs 20
    