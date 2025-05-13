# -*- coding: utf-8 -*-
"""
模型定义：BiLSTMAttention, CNN_BiLSTM, EnsembleModel
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 从 config 导入 ---
# (如果模型需要特定配置，可以在这里导入，但通常在 main.py 中实例化时传入)
# from config import INPUT_DIM, HIDDEN_DIM, NUM_LSTM_LAYERS, CNN_KERNEL_SIZE

class BiLSTMAttention(nn.Module):
    """BiLSTM + Attention 模型 (适用于预计算的全局特征)"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            bidirectional=True, batch_first=True, dropout=0.2 if num_layers > 1 else 0) # 添加 dropout
        # 注意力层，将 LSTM 输出映射到注意力分数
        self.attention_fc = nn.Linear(hidden_dim * 2, hidden_dim) # 映射到 hidden_dim
        self.attention_tanh = nn.Tanh()
        self.attention_vector = nn.Linear(hidden_dim, 1, bias=False) # 计算最终分数
        # 输出层
        self.output_fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入特征张量, shape [batch_size, feature_dim]

        Returns:
            torch.Tensor: 模型输出的 Logits, shape [batch_size, output_dim]
        """
        # 全局特征，虚拟序列长度为 1
        x = x.unsqueeze(1)  # [batch_size, 1, feature_dim]

        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, 1, hidden_dim * 2]

        # --- Attention 计算 (Bahdanau-style attention over the single time step) ---
        # 虽然只有一个时间步，但保持注意力结构
        attn_intermediate = self.attention_tanh(self.attention_fc(lstm_out)) # [batch, 1, hidden_dim]
        attn_scores = self.attention_vector(attn_intermediate).squeeze(-1) # [batch, 1]
        attn_weights = F.softmax(attn_scores, dim=1) # [batch, 1]

        # 计算上下文向量
        context = torch.sum(attn_weights.unsqueeze(-1) * lstm_out, dim=1) # [batch, hidden_dim * 2]
        # 对于 seq_len=1, context 等价于 lstm_out.squeeze(1)

        # --- 输出层 ---
        logits = self.output_fc(context)  # [batch_size, output_dim]
        # 不在此处应用 Sigmoid，交由 BCEWithLogitsLoss 处理
        return logits


class CNN_BiLSTM(nn.Module):
    """CNN + BiLSTM 模型 (适用于预计算的全局特征)"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 kernel_size: int = 3, num_layers: int = 1):
        super().__init__()
        # 1D CNN
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
                                kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.relu = nn.ReLU()
        self.dropout_cnn = nn.Dropout(0.25) # 添加 Dropout

        # BiLSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                            bidirectional=True, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.dropout_lstm = nn.Dropout(0.25) # 添加 Dropout

        # 输出层
        self.output_fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入特征张量, shape [batch_size, feature_dim]

        Returns:
            torch.Tensor: 模型输出的 Logits, shape [batch_size, output_dim]
        """
        # CNN 需要 [batch, channels, length]
        x = x.unsqueeze(2) # [batch, feature_dim, 1]

        cnn_out = self.conv1d(x) # [batch, hidden_dim, 1]
        cnn_out = self.relu(cnn_out)
        cnn_out = self.dropout_cnn(cnn_out)

        # LSTM 需要 [batch, seq_len, input_dim]
        cnn_out = cnn_out.permute(0, 2, 1) # [batch, 1, hidden_dim]

        lstm_out, (h_n, c_n) = self.lstm(cnn_out) # lstm_out: [batch, 1, hidden_dim * 2]

        # 取最后双向层的隐藏状态拼接作为输出
        fwd_last = h_n[-2, :, :]
        bwd_last = h_n[-1, :, :]
        final_lstm_output = torch.cat((fwd_last, bwd_last), dim=1) # [batch, hidden_dim * 2]
        final_lstm_output = self.dropout_lstm(final_lstm_output)

        logits = self.output_fc(final_lstm_output) # [batch, output_dim]
        # 不在此处应用 Sigmoid
        return logits


class EnsembleModel(nn.Module):
    """集成模型：加权平均 Logits"""
    def __init__(self, modelA: nn.Module, modelB: nn.Module, weightA: float = 0.5):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.weightA = weightA
        self.modelA.eval() # 确保子模型在评估模式
        self.modelB.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入特征张量, shape [batch_size, feature_dim]
                               (假设两个子模型都接受相同的输入格式)

        Returns:
            torch.Tensor: 集成后的 Logits, shape [batch_size, output_dim]
        """
        with torch.no_grad(): # 集成预测时不需要梯度
            # 假设两个模型都只需要 x 作为输入
            outA = self.modelA(x)
            outB = self.modelB(x)
        combined_logits = self.weightA * outA + (1 - self.weightA) * outB
        return combined_logits