import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    序列位置编码，用于编码事件在序列中的位置
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 创建一个足够长的位置编码表
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 使用正弦和余弦函数计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # 注册为缓冲区，但不作为模型参数
        self.register_buffer('pe', pe)

    def forward(self, seq_len, batch_size, device):
        """
        返回序列位置编码

        Args:
            seq_len: 序列长度
            batch_size: 批次大小
            device: 计算设备

        Returns:
            位置编码 [batch_size, seq_len, d_model]
        """
        pos_encoding = self.pe[:, :seq_len].expand(batch_size, -1, -1).to(device)
        return pos_encoding


class TimeEncoding(nn.Module):
    """
    连续时间值编码模块，将实际时间转换为高维表示
    """

    def __init__(self, d_model):
        super(TimeEncoding, self).__init__()
        self.d_model = d_model

        # 使用非线性投影将标量时间值投影到高维空间
        self.time_encoder = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

    def forward(self, time_values):
        """
        Args:
            time_values: 相对时间 [batch_size, seq_len]

        Returns:
            时间编码 [batch_size, seq_len, d_model]
        """
        # 将时间扩展为[batch_size, seq_len, 1]
        time_expanded = time_values.unsqueeze(-1)

        # 使用非线性投影编码时间
        time_encoding = self.time_encoder(time_expanded)

        return time_encoding


class ValueProcessor(nn.Module):
    """
    处理连续值（如lab检测值）的模块
    """

    def __init__(self, input_dim=1, hidden_dim=64, output_dim=128):
        super(ValueProcessor, self).__init__()
        self.value_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, values):
        """
        Args:
            values: 连续值 [batch_size, seq_len, 1] 或 [batch_size, seq_len]

        Returns:
            处理后的值特征 [batch_size, seq_len, output_dim]
        """
        # 确保输入有正确的形状
        if values.dim() == 2:
            values = values.unsqueeze(-1)  # [batch_size, seq_len] -> [batch_size, seq_len, 1]

        return self.value_mlp(values)


class TransformerBlock(nn.Module):
    """基础Transformer块，包含多头注意力和前馈网络"""

    def __init__(self, hidden_size, num_heads, intermediate_size, dropout_rate=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_rate, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout_rate)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_mask=None):
        # 多头注意力
        residual = x
        x_norm = self.norm1(x)
        x_attn, _ = self.attention(x_norm, x_norm, x_norm, key_padding_mask=attn_mask)
        x = residual + self.dropout(x_attn)

        # 前馈网络
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x_norm))

        return x
class ModalityEncoder(nn.Module):
    """为特定模态设计的Transformer编码器"""

    def __init__(self, hidden_size, num_layers, num_heads, intermediate_size, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size,
                num_heads,
                intermediate_size,
                dropout_rate
            ) for _ in range(num_layers)
        ])

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: 序列表示 [batch_size, seq_len, hidden_size]
            attention_mask: 原始掩码 [batch_size, seq_len]，1表示有效位置，0表示填充位置

        Returns:
            序列表示和CLS token表示
        """
        # 转换attention_mask格式为key_padding_mask，预期False为有效位置，True为填充位置
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()  # 对掩码取反
        else:
            key_padding_mask = None

        # 通过所有Transformer层
        for layer in self.layers:
            hidden_states = layer(hidden_states, key_padding_mask)

        # 返回整个序列和CLS token的表示
        return hidden_states, hidden_states[:, 0]


class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制
    允许一个模态的表示查询另一个模态的信息
    """

    def __init__(self, query_dim, key_dim, hidden_dim):
        super(CrossModalAttention, self).__init__()
        # 查询、键、值的线性映射
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, query_dim)

        # 其他参数
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key_value, key_mask=None):
        """
        参数:
            query: 查询表示 [batch_size, query_length, query_dim]
                  对于CLS查询，形状为 [batch_size, 1, query_dim]
            key_value: 被查询的表示 [batch_size, kv_length, key_dim]
            key_mask: 键值序列的掩码 [batch_size, kv_length]
                     True表示有效位置，False表示无效位置

        返回:
            context: 上下文向量 [batch_size, query_length, query_dim]
        """
        # 投影查询、键、值
        Q = self.query_proj(query)  # [batch_size, query_length, hidden_dim]
        K = self.key_proj(key_value)  # [batch_size, kv_length, hidden_dim]
        V = self.value_proj(key_value)  # [batch_size, kv_length, hidden_dim]

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (
                    self.hidden_dim ** 0.5)  # [batch_size, query_length, kv_length]

        # 应用掩码（如果提供）
        if key_mask is not None:
            # 扩展掩码以适应注意力分数的形状
            # [batch_size, 1, kv_length]
            expanded_mask = key_mask.unsqueeze(1)

            # 将掩码中的False（无效位置）对应的注意力分数设为一个非常大的负数
            scores = scores.masked_fill(~expanded_mask, -1e9)

        # 注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, query_length, kv_length]
        attn_weights = self.dropout(attn_weights)

        # 计算上下文向量
        context = torch.matmul(attn_weights, V)  # [batch_size, query_length, hidden_dim]

        # 最终输出投影
        context = self.output_proj(context)  # [batch_size, query_length, query_dim]

        return context