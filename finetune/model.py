from transformer import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class EHREmbeddings(nn.Module):
    """
    电子健康记录数据的嵌入层
    处理多种类型的医疗代码、时间信息、人口统计学特征和预计算的CLS嵌入
    """
    def __init__(self, config):
        super(EHREmbeddings, self).__init__()
        self.hidden_size = config.transformer_dim  # 隐藏层大小
        self.dx_vocab_size = len(config.dx_code_mapping)+1  # 4812
        self.rx_vocab_size = len(config.rx_code_mapping) +1 # 3158
        self.code_vocab_size = config.code_vocab_size  # 100001

        # 人口统计学特征编码器
        self.demographic_encoder = nn.Linear(70, config.transformer_dim)

        # 为不同类型的医疗代码创建嵌入表
        self.diagnosis_embeddings = nn.Embedding(
            self.dx_vocab_size, config.transformer_dim, padding_idx=0)
        self.medication_embeddings = nn.Embedding(
            self.rx_vocab_size, config.transformer_dim, padding_idx=0)
        self.drg_embeddings = nn.Embedding(
            self.code_vocab_size, config.transformer_dim, padding_idx=0)
        self.lab_embeddings = nn.Embedding(
            self.code_vocab_size, config.transformer_dim, padding_idx=0)
        self.discharge_code_embeddings = nn.Embedding(
            self.code_vocab_size, config.transformer_dim, padding_idx=0)
        self.radiology_code_embeddings = nn.Embedding(
            self.code_vocab_size, config.transformer_dim, padding_idx=0)

        # 为预计算的CLS嵌入添加投影层
        self.discharge_summary_projection = nn.Linear(768, config.transformer_dim)
        self.radiology_report_projection = nn.Linear(768, config.transformer_dim)

        # 时间编码（连续时间值）
        self.time_encoding = TimeEncoding(config.transformer_dim)

        # 序列位置编码
        self.position_encoding = PositionalEncoding(config.transformer_dim)

        # lab值处理器
        self.lab_value_processor = ValueProcessor(
            input_dim=1,
            hidden_dim=config.transformer_dim // 2,
            output_dim=config.transformer_dim
        )

        # 规范化和dropout层
        self.layer_norm = nn.LayerNorm(config.transformer_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward_code_embeddings(self, codes, times, embedding_table, mask=None):
        """
        处理医疗代码和对应的时间

        Args:
            codes: 代码索引 [batch_size, seq_len]
            times: 相对时间 [batch_size, seq_len]
            embedding_table: 使用的嵌入表
            mask: 有效位置掩码 [batch_size, seq_len]

        Returns:
            嵌入向量 [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = codes.shape

        # 获取代码嵌入
        code_emb = embedding_table(codes)

        # 获取时间编码
        time_emb = self.time_encoding(times)

        # 获取位置编码
        pos_emb = self.position_encoding(seq_len, batch_size, codes.device)

        # 合并所有嵌入
        embeddings = code_emb + time_emb + pos_emb

        # 应用规范化和dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # 如果提供了掩码，将填充位置的嵌入置为0
        if mask is not None:
            embeddings = embeddings * mask.unsqueeze(-1)

        return embeddings

    def forward_lab_embeddings(self, codes, times, values, mask=None):
        """
        处理实验室检测数据，包括代码、时间和值

        Args:
            codes: lab代码索引 [batch_size, seq_len]
            times: 相对时间 [batch_size, seq_len]
            values: lab值 [batch_size, seq_len]
            mask: 有效位置掩码 [batch_size, seq_len]

        Returns:
            lab嵌入 [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = codes.shape

        # 获取lab代码嵌入
        code_emb = self.lab_embeddings(codes)

        # 处理lab值
        value_emb = self.lab_value_processor(values)

        # 获取时间编码
        time_emb = self.time_encoding(times)

        # 获取位置编码
        pos_emb = self.position_encoding(seq_len, batch_size, codes.device)

        # 合并所有嵌入
        embeddings = code_emb + value_emb + time_emb + pos_emb

        # 应用规范化和dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # 如果提供了掩码，将填充位置的嵌入置为0
        if mask is not None:
            embeddings = embeddings * mask.unsqueeze(-1)

        return embeddings

    def forward_cls_embeddings(self, embeddings, modality):
        """
        处理预计算的CLS嵌入（discharge_summary 或 radiology_report）

        Args:
            embeddings: 预计算的CLS嵌入 [batch_size, num_notes, 768]
            modality: 'discharge_summary' 或 'radiology_report'

        Returns:
            投影后的CLS嵌入 [batch_size, hidden_size]
        """
        # 对所有note的CLS取平均，得到单一CLS表示
        cls_emb = torch.mean(embeddings, dim=1)  # [batch_size, 768]

        # 根据模态选择投影层
        if modality == 'discharge_summary':
            projected_emb = self.discharge_summary_projection(cls_emb)
        elif modality == 'radiology_report':
            projected_emb = self.radiology_report_projection(cls_emb)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        # 应用规范化
        projected_emb = self.layer_norm(projected_emb)
        projected_emb = self.dropout(projected_emb)

        return projected_emb

    def forward(self, batch):
        """
        处理包含多种模态数据的批次

        Args:
            batch: 包含多种模态的数据批次

        Returns:
            处理后的嵌入字典，按模态组织；掩码字典
        """
        embeddings = {}
        masks = {}

        # 处理人口统计学特征
        if 'demographic' in batch:
            embeddings['demographic'] = self.demographic_encoder(batch['demographic'])

        # 处理诊断代码
        if 'diagnosis' in batch:
            masks['diagnosis'] = batch['diagnosis']['mask'] if 'mask' in batch['diagnosis'] else None
            embeddings['diagnosis'] = self.forward_code_embeddings(
                batch['diagnosis']['codes'],
                batch['diagnosis']['times'],
                self.diagnosis_embeddings,
                masks['diagnosis']
            )

        # 处理药物代码
        if 'medication' in batch:
            masks['medication'] = batch['medication']['mask'] if 'mask' in batch['medication'] else None
            embeddings['medication'] = self.forward_code_embeddings(
                batch['medication']['codes'],
                batch['medication']['times'],
                self.medication_embeddings,
                masks['medication']
            )

        # 处理DRG代码
        if 'drg' in batch:
            masks['drg'] = batch['drg']['mask'] if 'mask' in batch['drg'] else None
            embeddings['drg'] = self.forward_code_embeddings(
                batch['drg']['codes'],
                batch['drg']['times'],
                self.drg_embeddings,
                masks['drg']
            )

        # 处理出院摘要代码（作为token序列）
        if 'discharge_codes' in batch:
            masks['discharge_codes'] = batch['discharge_codes']['mask'] if 'mask' in batch['discharge_codes'] else None
            embeddings['discharge_codes'] = self.forward_code_embeddings(
                batch['discharge_codes']['codes'],
                batch['discharge_codes']['times'],
                self.discharge_code_embeddings,
                masks['discharge_codes']
            )

        # 处理放射学报告代码
        if 'radiology_codes' in batch:
            masks['radiology_codes'] = batch['radiology_codes']['mask'] if 'mask' in batch['radiology_codes'] else None
            embeddings['radiology_codes'] = self.forward_code_embeddings(
                batch['radiology_codes']['codes'],
                batch['radiology_codes']['times'],
                self.radiology_code_embeddings,
                masks['radiology_codes']
            )

        # 处理实验室检测事件
        if 'lab' in batch:
            masks['lab'] = batch['lab']['mask'] if 'mask' in batch['lab'] else None
            embeddings['lab'] = self.forward_lab_embeddings(
                batch['lab']['codes'],
                batch['lab']['times'],
                batch['lab']['values'],
                masks['lab']
            )

        # 处理出院摘要CLS嵌入
        if 'discharge_summary' in batch and batch['discharge_summary']['embeddings'] is not None:
            embeddings['discharge_summary'] = self.forward_cls_embeddings(
                batch['discharge_summary']['embeddings'],
                'discharge_summary'
            )
            masks['discharge_summary'] = None  # CLS没有序列掩码

        # 处理放射学报告CLS嵌入
        if 'radiology_report' in batch and batch['radiology_report']['embeddings'] is not None:
            embeddings['radiology_report'] = self.forward_cls_embeddings(
                batch['radiology_report']['embeddings'],
                'radiology_report'
            )
            masks['radiology_report'] = None  # CLS没有序列掩码

        return embeddings, masks

import torch
from utils import  focal_loss

class EHR_Model(nn.Module):
    """多模态电子健康记录预训练模型"""
    def __init__(self, args):
        super().__init__()
        self.transformer_dim = args.transformer_dim
        self.dx_vocab_size = len(args.dx_code_mapping)+1  # 4812
        self.rx_vocab_size = len(args.rx_code_mapping)+1  # 3158
        self.code_vocab_size = args.code_vocab_size  # 100001

        # 嵌入层
        self.embeddings = EHREmbeddings(args)

        # 模态编码器（只对序列模态使用）
        intermediate_size = args.transformer_dim * 4
        self.encoders = nn.ModuleDict({
            'diagnosis': ModalityEncoder(
                args.transformer_dim, args.transformer_num_layers,
                args.transformer_num_head, intermediate_size, args.dropout
            ),
            'medication': ModalityEncoder(
                args.transformer_dim, args.transformer_num_layers,
                args.transformer_num_head, intermediate_size, args.dropout
            ),
            'discharge_codes': ModalityEncoder(
                args.transformer_dim, args.transformer_num_layers,
                args.transformer_num_head, intermediate_size, args.dropout
            ),
        })

        # 跨模态注意力机制
        self.dx2text = CrossModalAttention(args.transformer_dim, args.transformer_dim, args.transformer_dim)
        self.rx2text = CrossModalAttention(args.transformer_dim, args.transformer_dim, args.transformer_dim)
        self.text2dx = CrossModalAttention(args.transformer_dim, args.transformer_dim, args.transformer_dim)
        self.text2rx = CrossModalAttention(args.transformer_dim, args.transformer_dim, args.transformer_dim)

        # LayerNorm 用于残差连接
        self.layer_norm = nn.LayerNorm(args.transformer_dim)

        # 预测头
        self.dx2dx_head = nn.Linear(args.transformer_dim, self.dx_vocab_size)  # 4812
        self.rx2dx_head = nn.Linear(args.transformer_dim, self.dx_vocab_size)  # 4812
        self.dx2rx_head = nn.Linear(args.transformer_dim, self.rx_vocab_size)  # 3158
        self.rx2rx_head = nn.Linear(args.transformer_dim, self.rx_vocab_size)  # 3158
        self.text2dx_head = nn.Linear(args.transformer_dim, self.dx_vocab_size)  # 4812
        self.text2rx_head = nn.Linear(args.transformer_dim, self.rx_vocab_size)  # 3158

    def forward(self, batch):
        """
        前向传播

        Args:
            batch: 包含多种模态的数据批次

        Returns:
            total_loss: 六个预训练任务的总损失
        """
        # 获取嵌入和掩码
        embeddings, masks = self.embeddings(batch)

        # 编码序列模态并提取CLS
        encoded_sequences = {}
        encoded_cls = {}
        for modality in ['diagnosis', 'medication', 'discharge_codes']:
            if modality in embeddings:
                seq, cls = self.encoders[modality](embeddings[modality], masks.get(modality, None))
                encoded_sequences[modality] = seq
                encoded_cls[modality] = cls

        # discharge_summary 直接使用CLS嵌入
        if 'discharge_summary' in embeddings:
            encoded_cls['discharge_summary'] = embeddings['discharge_summary']

        # 跨模态注意力融合
        enhanced_representations = {}

        # 1. 诊断与discharge_codes交互
        if 'diagnosis' in encoded_cls and 'discharge_codes' in encoded_sequences:
            dx_bert_att = self.dx2text(
                encoded_cls['diagnosis'].unsqueeze(1),
                encoded_sequences['discharge_codes'],
                masks.get('discharge_codes', None)
            ).squeeze(1)
            enhanced_dx = self.layer_norm(encoded_cls['diagnosis'] + dx_bert_att)
            enhanced_representations['enhanced_dx'] = enhanced_dx

        # 2. 药物与discharge_codes交互
        if 'medication' in encoded_cls and 'discharge_codes' in encoded_sequences:
            rx_bert_att = self.rx2text(
                encoded_cls['medication'].unsqueeze(1),
                encoded_sequences['discharge_codes'],
                masks.get('discharge_codes', None)
            ).squeeze(1)
            enhanced_rx = self.layer_norm(encoded_cls['medication'] + rx_bert_att)
            enhanced_representations['enhanced_rx'] = enhanced_rx

        # 3. discharge_summary与诊断/药物交互
        if 'discharge_summary' in encoded_cls:
            if 'diagnosis' in encoded_sequences:
                text_dx_att = self.text2dx(
                    encoded_cls['discharge_summary'].unsqueeze(1),
                    encoded_sequences['diagnosis'],
                    masks.get('diagnosis', None)
                ).squeeze(1)
                enhanced_text_dx = self.layer_norm(encoded_cls['discharge_summary'] + text_dx_att)
                enhanced_representations['enhanced_text_dx'] = enhanced_text_dx

            if 'medication' in encoded_sequences:
                text_rx_att = self.text2rx(
                    encoded_cls['discharge_summary'].unsqueeze(1),
                    encoded_sequences['medication'],
                    masks.get('medication', None)
                ).squeeze(1)
                enhanced_text_rx = self.layer_norm(encoded_cls['discharge_summary'] + text_rx_att)
                enhanced_representations['enhanced_text_rx'] = enhanced_text_rx

        # 预训练任务
        losses = {}
        correct_counts = {}

        # 1. dx2dx: 用增强后的诊断CLS预测诊断代码
        if 'enhanced_dx' in enhanced_representations and 'original_codes' in batch and 'diagnosis' in batch['original_codes']:
            dx2dx_pred = self.dx2dx_head(enhanced_representations['enhanced_dx'])
            dx_one_hot = indices_to_one_hot(batch['original_codes']['diagnosis'], self.dx_vocab_size)
            losses['dx2dx'] = focal_loss(dx2dx_pred, dx_one_hot, gamma=2.0, reduction='mean')  # 使用 Focal Loss

            # 计算预测概率并获取 Top-15
            dx_probs = torch.sigmoid(dx2dx_pred)
            top15_probs, top15_indices = torch.topk(dx_probs, k=15, dim=1)

            # 按代码大小排序
            sorted_indices, sort_order = torch.sort(top15_indices, dim=1)
            sorted_probs = torch.gather(top15_probs, dim=1, index=sort_order)

            # 获取真实标签的索引
            true_indices = batch['original_codes']['diagnosis']

            # 计算整个批次的正确预测数
            batch_size = dx2dx_pred.size(0)
            correct_counts['dx2dx'] = 0
            for i in range(batch_size):  # 遍历整个批次
                true_set = set(true_indices[i][true_indices[i] > 0].tolist())
                pred_set = set(sorted_indices[i].tolist())
                correct = len(true_set & pred_set)
                correct_counts['dx2dx'] += correct

            # 只打印第一个样本的 Top-15 预测
            print("\n=== dx2dx Top-15 Predictions ===")
            for i in range(min(batch_size, 1)):  # 只打印第一个样本
                print(f"Sample {i}:")
                print(f"  Top-15 Predicted Codes (sorted by code): {sorted_indices[i].tolist()}")
                print(f"  Top-15 Probabilities (sorted by code): {sorted_probs[i].tolist()}")
                print(f"  True Codes: {true_indices[i][true_indices[i] > 0].tolist()}")
                print(
                    f"  Correct in Top-15: {len(set(true_indices[i][true_indices[i] > 0].tolist()) & set(sorted_indices[i].tolist()))}/{len(set(true_indices[i][true_indices[i] > 0].tolist()))}")

            print(f"Batch Total Correct (dx2dx): {correct_counts['dx2dx']} / {batch_size * 15}")

        # 2. rx2dx: 用增强后的药物CLS预测诊断代码
        if 'enhanced_rx' in enhanced_representations and 'original_codes' in batch and 'diagnosis' in batch['original_codes']:
            rx2dx_pred = self.rx2dx_head(enhanced_representations['enhanced_rx'])
            dx_one_hot = indices_to_one_hot(batch['original_codes']['diagnosis'], self.dx_vocab_size)
            losses['rx2dx'] = focal_loss(rx2dx_pred, dx_one_hot, gamma=2.0, reduction='mean')  # 使用 Focal Loss

        # 3. dx2rx: 用增强后的诊断CLS预测药物代码
        if 'enhanced_dx' in enhanced_representations and 'original_codes' in batch and 'medication' in batch['original_codes']:
            dx2rx_pred = self.dx2rx_head(enhanced_representations['enhanced_dx'])
            rx_one_hot = indices_to_one_hot(batch['original_codes']['medication'], self.rx_vocab_size)
            losses['dx2rx'] = focal_loss(dx2rx_pred, rx_one_hot, gamma=2.0, reduction='mean')  # 使用 Focal Loss

        # 4. rx2rx: 用增强后的药物CLS预测药物代码
        if 'enhanced_rx' in enhanced_representations and 'original_codes' in batch and 'medication' in batch['original_codes']:
            rx2rx_pred = self.rx2rx_head(enhanced_representations['enhanced_rx'])
            rx_one_hot = indices_to_one_hot(batch['original_codes']['medication'], self.rx_vocab_size)
            losses['rx2rx'] = focal_loss(rx2rx_pred, rx_one_hot, gamma=2.0, reduction='mean')  # 使用 Focal Loss

        # 5. text2dx: 用增强后的discharge_summary CLS预测诊断代码
        if 'enhanced_text_dx' in enhanced_representations and 'original_codes' in batch and 'diagnosis' in batch['original_codes']:
            text2dx_pred = self.text2dx_head(enhanced_representations['enhanced_text_dx'])
            dx_one_hot = indices_to_one_hot(batch['original_codes']['diagnosis'], self.dx_vocab_size)
            losses['text2dx'] = focal_loss(text2dx_pred, dx_one_hot, gamma=2.0, reduction='mean')  # 使用 Focal Loss

        # 6. text2rx: 用增强后的discharge_summary CLS预测药物代码
        if 'enhanced_text_rx' in enhanced_representations and 'original_codes' in batch and 'medication' in batch['original_codes']:
            text2rx_pred = self.text2rx_head(enhanced_representations['enhanced_text_rx'])
            rx_one_hot = indices_to_one_hot(batch['original_codes']['medication'], self.rx_vocab_size)
            losses['text2rx'] = focal_loss(text2rx_pred, rx_one_hot, gamma=2.0, reduction='mean')  # 使用 Focal Loss

        # 总损失
        total_loss = sum(losses.values()) if losses else torch.tensor(0.0, device=next(self.parameters()).device)

        return total_loss

def indices_to_one_hot(indices, vocab_size):
    """
    将索引形式的标签转换为one-hot编码

    Args:
        indices: 索引张量 [batch, seq_len]
        vocab_size: 词汇表大小（dx_vocab_size 或 rx_vocab_size）

    Returns:
        one_hot: one-hot编码 [batch, vocab_size]
    """
    device = indices.device
    batch_size = indices.size(0)
    one_hot = torch.zeros(batch_size, vocab_size, device=device)

    for i in range(batch_size):
        # 过滤掉填充的零和掩码标记
        valid_indices = indices[i][(indices[i] > 0) & (indices[i] < vocab_size)]
        if len(valid_indices) > 0:
            one_hot[i].scatter_(0, valid_indices, 1.0)

    return one_hot