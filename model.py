from transformer import *


class TrimodalTransformerEncoder(nn.Module):
    """
    基于MBT的三模态Transformer编码器
    适配医疗代码、注释嵌入和实验室检测值三种模态
    """

    def __init__(self,
                 batch_size: int,
                 bottlenecks_n: int,
                 fusion_startidx: int,
                 n_layers: int,
                 n_head: int,
                 d_model: int,
                 d_ff: int,
                 dropout: float = 0.1,
                 pe_maxlen: int = 10000,
                 use_pe: list = [True, True, True],
                 mask: list = [True, True, True],
                 device=None):
        super(TrimodalTransformerEncoder, self).__init__()

        self.use_pe = use_pe
        self.n_modality = 3  # 医疗代码模态、文本模态和实验室检测模态
        self.fusion_idx = fusion_startidx
        self.n_layers = n_layers
        self.d_model = d_model
        self.bottlenecks_n = bottlenecks_n
        self.mask = mask
        self.device = device

        self.idx_order = torch.arange(0, batch_size).to(torch.long)
        if self.device:
            self.idx_order = self.idx_order.to(self.device)

        # CLS tokens
        self.cls_token_per_modality = nn.ParameterList(
            [nn.Parameter(torch.randn(1, 1, d_model)) for _ in range(self.n_modality)])

        # Bottleneck tokens
        self.bottlenecks = nn.Parameter(torch.randn(1, bottlenecks_n, d_model))

        # Layer normalization
        self.layer_norms_in = nn.ModuleList([LayerNorm(d_model) for _ in range(self.n_modality)])
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        # Transformer layers stacks
        self.layer_stacks = nn.ModuleList([
            nn.ModuleList([
                TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=n_head,
                    d_ff=d_ff,
                    dropout_p=dropout
                ) for _ in range(self.n_modality)
            ]) for _ in range(n_layers)
        ])

    def forward(self, enc_outputs, lengths=None, return_attns=False, fusion_idx=None, missing=None):
        """
        前向传播函数

        参数:
        enc_outputs: 模态嵌入列表 [medical_emb, note_emb, lab_emb]
        lengths: 每个模态序列的实际长度 [medical_lengths, note_lengths, lab_lengths]
        return_attns: 是否返回注意力权重
        fusion_idx: 开始融合的层索引
        missing: 缺失模态标记
        """
        batch_size = enc_outputs[0].size(0)

        # 复制CLS tokens到每个批次
        cls_token_per_modality = [cls_token.repeat(batch_size, 1, 1) for cls_token in
                                  self.cls_token_per_modality]

        # 复制bottleneck tokens到每个批次
        bottlenecks = self.bottlenecks.repeat(batch_size, 1, 1)

        # 在序列前添加CLS token
        enc_inputs = [torch.cat([cls_token_per_modality[idx], enc_output], dim=1) for idx, enc_output in
                      enumerate(enc_outputs)]

        # 创建attention masks
        self_attn_masks = []
        bottleneck_self_attn_masks = []

        # 为每个模态创建掩码
        for n_modal in range(self.n_modality):
            if self.mask[n_modal] and lengths is not None:
                # 增加1以适应CLS token
                modal_lengths = lengths[n_modal] + 1

                # 创建掩码张量 (batch_size, seq_len, seq_len)
                seq_len = enc_inputs[n_modal].size(1)
                mask = torch.zeros(batch_size, seq_len, seq_len, device=self.device).bool()

                # 对于每个样本
                for i in range(batch_size):
                    # 创建一个与序列长度匹配的掩码
                    seq_mask = torch.zeros(seq_len).bool()
                    # 设置实际数据部分为True (包括CLS token)
                    seq_mask[:modal_lengths[i]] = True

                    # 创建注意力掩码: ~(seq_mask.unsqueeze(1) & seq_mask.unsqueeze(0))
                    # 这里取反是因为在我们的attention中，True代表被掩盖的位置
                    attn_mask = ~(seq_mask.unsqueeze(1) & seq_mask.unsqueeze(0))
                    mask[i] = attn_mask

                if self.device:
                    mask = mask.to(self.device)

                self_attn_masks.append(mask)
            else:
                self_attn_masks.append(None)

        # 如果指定了fusion_idx，使用它
        if fusion_idx is not None:
            self.fusion_idx = fusion_idx

        # 添加位置编码
        enc_outputs = []
        for idx, pe_bool in enumerate(self.use_pe):
            if pe_bool:
                # 获取位置编码
                position_enc = self.positional_encoding(enc_inputs[idx].size(1))
                # 应用位置编码
                enc_outputs.append(self.dropout(
                    self.layer_norms_in[idx](enc_inputs[idx]) + position_enc
                ))
            else:
                enc_outputs.append(self.dropout(
                    self.layer_norms_in[idx](enc_inputs[idx])
                ))

        # 通过Transformer层
        for idx, enc_layers in enumerate(self.layer_stacks):
            enc_inputs = list(enc_outputs)
            enc_outputs = []

            # 前fusion_idx层没有模态间交互
            if idx < self.fusion_idx:
                for modal_idx, enc_layer in enumerate(enc_layers):
                    enc_output, _ = enc_layer(enc_inputs[modal_idx], self_attn_masks[modal_idx])
                    enc_outputs.append(enc_output)

            # 之后的层使用bottlenecks进行模态间交互
            else:
                bottleneck_outputs = []

                for modal_idx, enc_layer in enumerate(enc_layers):
                    # 将bottlenecks添加到输入序列
                    b_enc_output = torch.cat([bottlenecks, enc_inputs[modal_idx]], dim=1)

                    # 为带bottlenecks的序列创建mask
                    if len(bottleneck_self_attn_masks) < self.n_modality:
                        if self.mask[modal_idx] and lengths is not None:
                            # 增加1以适应CLS token
                            modal_lengths = lengths[modal_idx] + 1

                            # 创建掩码张量 (batch_size, bottleneck+seq_len, bottleneck+seq_len)
                            b_seq_len = b_enc_output.size(1)
                            b_mask = torch.zeros(batch_size, b_seq_len, b_seq_len, device=self.device).bool()

                            # 对于每个样本
                            for i in range(batch_size):
                                # 创建一个与序列长度匹配的掩码
                                bottleneck_mask = torch.ones(self.bottlenecks_n).bool()  # bottlenecks总是可见
                                seq_mask = torch.zeros(enc_inputs[modal_idx].size(1)).bool()
                                seq_mask[:modal_lengths[i]] = True  # 包括CLS token

                                # 合并掩码
                                full_mask = torch.cat([bottleneck_mask, seq_mask])

                                # 创建注意力掩码
                                attn_mask = ~(full_mask.unsqueeze(1) & full_mask.unsqueeze(0))
                                b_mask[i] = attn_mask

                            if self.device:
                                b_mask = b_mask.to(self.device)

                            bottleneck_self_attn_masks.append(b_mask)
                        else:
                            bottleneck_self_attn_masks.append(None)

                    # 通过Transformer层
                    enc_output, _ = enc_layer(b_enc_output, bottleneck_self_attn_masks[modal_idx])

                    # 提取bottleneck输出和序列输出
                    bottleneck_outputs.append(enc_output[:, :self.bottlenecks_n, :])
                    enc_output = enc_output[:, self.bottlenecks_n:, :]
                    enc_outputs.append(enc_output)

                # 更新bottlenecks，考虑不同的模态缺失情况
                bottleneck_outputs_stack = torch.stack(
                    bottleneck_outputs)  # [n_modality, batch_size, bottlenecks_n, d_model]

                # 创建不同组合的bottleneck平均值
                bottlenecks_tri_mean = torch.mean(bottleneck_outputs_stack,
                                                  dim=0)  # 三模态平均 [batch_size, bottlenecks_n, d_model]

                # 如果存在缺失模态，需要特殊处理
                if missing is not None:
                    # 医疗代码+注释
                    med_note_indices = (missing[:, 0] == 1) & (missing[:, 1] == 1) & (missing[:, 2] == 0)
                    # 医疗代码+实验室
                    med_lab_indices = (missing[:, 0] == 1) & (missing[:, 1] == 0) & (missing[:, 2] == 1)
                    # 注释+实验室
                    note_lab_indices = (missing[:, 0] == 0) & (missing[:, 1] == 1) & (missing[:, 2] == 1)
                    # 仅医疗代码
                    only_med_indices = (missing[:, 0] == 1) & (missing[:, 1] == 0) & (missing[:, 2] == 0)
                    # 仅注释
                    only_note_indices = (missing[:, 0] == 0) & (missing[:, 1] == 1) & (missing[:, 2] == 0)
                    # 仅实验室
                    only_lab_indices = (missing[:, 0] == 0) & (missing[:, 1] == 0) & (missing[:, 2] == 1)

                    # 初始化新的bottlenecks（保持原始值）
                    new_bottlenecks = bottlenecks.clone()

                    # 处理所有模态都存在的情况（三模态平均）
                    all_present_indices = (missing.sum(dim=1) == 3)
                    if all_present_indices.any():
                        new_bottlenecks[all_present_indices] = bottlenecks_tri_mean[all_present_indices]

                    # 处理医疗代码+注释
                    if med_note_indices.any():
                        med_note_mean = torch.mean(torch.stack([
                            bottleneck_outputs_stack[0, med_note_indices],
                            bottleneck_outputs_stack[1, med_note_indices]
                        ]), dim=0)
                        new_bottlenecks[med_note_indices] = med_note_mean

                    # 处理医疗代码+实验室
                    if med_lab_indices.any():
                        med_lab_mean = torch.mean(torch.stack([
                            bottleneck_outputs_stack[0, med_lab_indices],
                            bottleneck_outputs_stack[2, med_lab_indices]
                        ]), dim=0)
                        new_bottlenecks[med_lab_indices] = med_lab_mean

                    # 处理注释+实验室
                    if note_lab_indices.any():
                        note_lab_mean = torch.mean(torch.stack([
                            bottleneck_outputs_stack[1, note_lab_indices],
                            bottleneck_outputs_stack[2, note_lab_indices]
                        ]), dim=0)
                        new_bottlenecks[note_lab_indices] = note_lab_mean

                    # 处理单模态情况
                    if only_med_indices.any():
                        new_bottlenecks[only_med_indices] = bottleneck_outputs_stack[0, only_med_indices]
                    if only_note_indices.any():
                        new_bottlenecks[only_note_indices] = bottleneck_outputs_stack[1, only_note_indices]
                    if only_lab_indices.any():
                        new_bottlenecks[only_lab_indices] = bottleneck_outputs_stack[2, only_lab_indices]

                    # 更新bottlenecks
                    bottlenecks = new_bottlenecks
                else:
                    # 如果没有缺失信息，简单地使用三模态平均
                    bottlenecks = bottlenecks_tri_mean

        return enc_outputs, 0


class EHR_MBT_Model(nn.Module):
    def __init__(self, args,code_dict):
        super().__init__()
        self.args = args

        # 配置
        self.output_dim = 1  # 二分类任务（再入院预测）
        self.num_layers = args.transformer_num_layers
        self.num_heads = args.transformer_num_head
        self.model_dim = args.transformer_dim
        self.dropout = args.dropout
        self.device = args.device
        self.n_modality = 3  # 医疗代码模态、文本模态和实验室检测模态
        self.bottlenecks_n = 4

        # 定义类型ID常量
        self.TYPE_DIAGNOSIS = 0
        self.TYPE_MEDICATION = 1
        self.TYPE_DRG = 2
        self.TYPE_DISCHARGE = 3
        self.TYPE_RADIOLOGY = 4
        self.TYPE_LAB_BASE = 100  # lab类型的基础ID（实际ID是100+lab_code_index）
        self.TYPE_PADDING = 99

        # 激活函数选择
        activation = 'relu'
        self.activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['prelu', nn.PReLU()],
            ['relu', nn.ReLU(inplace=True)],
            ['tanh', nn.Tanh()],
            ['sigmoid', nn.Sigmoid()],
            ['leaky_relu', nn.LeakyReLU(0.2)],
            ['elu', nn.ELU()]
        ])

        # 创建可学习的注意力层来计算模态间权重
        self.cls_fusion_attention = nn.Sequential(
            nn.Linear(self.model_dim, 3),  # 输出三个权重，对应三个模态
            nn.Softmax(dim=1)  # 使用softmax确保权重总和为1
        ).to(self.device)


        if code_dict is not None:
            # 获取最大索引和嵌入维度
            max_code_index = max(code_dict['index']) + 1
            embedding_dim = len(code_dict['bge_embedding'].iloc[0])

            # 创建一个嵌入矩阵，初始化为零
            embeddings_matrix = torch.zeros((max_code_index, embedding_dim))

            # 填充预训练的嵌入
            for i in range(len(code_dict)):
                idx = code_dict['index'].iloc[i]
                embeddings_matrix[idx] = torch.tensor(code_dict['bge_embedding'].iloc[i])

            # 使用这个矩阵创建嵌入层
            self.code_embedding = nn.Embedding.from_pretrained(embeddings_matrix, padding_idx=0, freeze=True)

            self.code_projection = nn.Linear(embedding_dim, self.model_dim)

        else:
            # 原来的嵌入层
            self.code_embedding = nn.Embedding(args.code_vocab_size, self.model_dim)
            self.code_projection = nn.Identity()

        # 共享的时间编码器
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.model_dim),
            LayerNorm(self.model_dim),
            nn.ReLU(inplace=True),
        )
        # 共享的类型编码器 (最大支持1000种类型，包括100+lab_code_index)
        self.type_embedding = nn.Embedding(12332, self.model_dim, padding_idx=99)
        # 人口统计学编码器 (维度为70)
        self.demo_embedding = nn.Sequential(
            nn.Linear(70, self.model_dim),
            LayerNorm(self.model_dim),
            nn.ReLU(inplace=True),
        )
        # 注释嵌入的变换层 (假设BioBERT嵌入是768维)
        self.note_transform = nn.Linear(768, self.model_dim)

        self.lab_value_transform = nn.Sequential(
            nn.Linear(1, self.model_dim),
            LayerNorm(self.model_dim),
            nn.ReLU(inplace=True),
        )

        ##### 融合部分
        self.fusion_transformer = TrimodalTransformerEncoder(
            batch_size=args.batch_size,
            bottlenecks_n=self.bottlenecks_n,
            fusion_startidx=args.fusion_startIdx,
            n_layers=self.num_layers,
            n_head=self.num_heads,
            d_model=self.model_dim,
            d_ff=self.model_dim * 4,
            dropout=self.dropout,
            pe_maxlen=2500,
            use_pe=[True, True, True],  # 三个模态都使用位置编码
            mask=[True, True, True],  # 三个模态都使用掩码
            device=self.device
        )

        ##### 分类器
        self.layer_norm = LayerNorm(self.model_dim)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.model_dim, out_features=self.model_dim, bias=True),
            LayerNorm(self.model_dim),
            self.activations[activation],
            nn.Linear(in_features=self.model_dim, out_features=self.output_dim, bias=True)
        )

    def forward(self, batch, missing=None):
        """
        前向传播函数

        参数:
        batch: 包含以下键的字典:
            - demographic: 人口统计学特征 [batch_size, 70]
            - medical: {
                'codes': [batch_size, seq_len],
                'times': [batch_size, seq_len],
                'types': [batch_size, seq_len],
                'mask': [batch_size, seq_len],
                'lengths': [batch_size]
              }
            - notes: {
                'embeddings': [batch_size, seq_len, 768],
                'times': [batch_size, seq_len],
                'types': [batch_size, seq_len],
                'mask': [batch_size, seq_len],
                'lengths': [batch_size]
              }
            - labs: {
                'codes': [batch_size, seq_len],
                'times': [batch_size, seq_len],
                'values': [batch_size, seq_len],
                'types': [batch_size, seq_len],
                'mask': [batch_size, seq_len],
                'lengths': [batch_size]
              }
            - label: [batch_size]
        missing: 缺失模态标记，格式为 [batch_size, 3]，其中1表示模态存在，0表示缺失

        返回:
        output: 再入院预测概率 [batch_size, 1]
        """
        # 获取批次大小
        batch_size = batch['demographic'].size(0)

        # 处理人口统计学特征
        demo_emb = self.demo_embedding(batch['demographic'])

        # ===== 处理医疗代码数据 =====
        medical_codes = batch['medical']['codes']
        medical_times = batch['medical']['times'].unsqueeze(-1)  # 添加特征维度
        medical_types = batch['medical']['types']
        medical_mask = batch['medical']['mask']
        medical_lengths = batch['medical']['lengths']

        # 获取各种嵌入
        code_emb = self.code_embedding(medical_codes)  # 使用预训练的嵌入
        code_emb = self.code_projection(code_emb)  # 投影到模型维度（如果需要）
        time_emb = self.time_embedding(medical_times)
        type_emb = self.type_embedding(medical_types)

        # 合并嵌入
        medical_emb = code_emb + time_emb + type_emb

        # ===== 处理临床文本数据 =====
        if 'notes' in batch and batch['notes']['embeddings'] is not None:
            note_emb = batch['notes']['embeddings']
            note_times = batch['notes']['times'].unsqueeze(-1)  # 添加特征维度
            note_types = batch['notes']['types']
            note_mask = batch['notes']['mask']
            note_lengths = batch['notes']['lengths']

            # 转换嵌入维度并添加时间和类型信息
            note_value_emb = self.note_transform(note_emb)  # 转换维度
            note_time_emb = self.time_embedding(note_times)  # 时间嵌入
            note_type_emb = self.type_embedding(note_types)  # 类型嵌入

            # 合并嵌入
            note_emb = note_value_emb + note_time_emb + note_type_emb
        else:
            # 创建空的note嵌入和长度
            note_emb = torch.zeros(batch_size, 1, self.model_dim, device=self.device)
            note_lengths = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # ===== 处理实验室检测数据 - 新增部分 =====
        if 'labs' in batch and batch['labs']['types'] is not None:
            #lab_codes = batch['labs']['codes']
            lab_times = batch['labs']['times'].unsqueeze(-1)  # 添加特征维度
            lab_values = batch['labs']['values'].unsqueeze(-1)  # 添加特征维度
            lab_types = batch['labs']['types']
            lab_mask = batch['labs']['mask']
            lab_lengths = batch['labs']['lengths']

            # 获取各种嵌入
            #lab_code_emb = self.lab_code_embedding(lab_codes)  # lab代码嵌入
            lab_time_emb = self.time_embedding(lab_times)  # 时间嵌入
            lab_value_emb = self.lab_value_transform(lab_values)  # lab值嵌入
            lab_type_emb = self.type_embedding(lab_types)  # 类型嵌入

            # 合并嵌入
            lab_emb =  lab_time_emb + lab_value_emb + lab_type_emb
        else:
            # 创建空的lab嵌入和长度
            lab_emb = torch.zeros(batch_size, 1, self.model_dim, device=self.device)
            lab_lengths = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # ===== 处理缺失模态情况 =====
        # 如果未提供missing参数，则自动检测缺失情况
        if missing is None:
            missing = torch.ones(batch_size, 3, dtype=torch.long, device=self.device)

            # 检测医疗代码模态是否有效
            for i in range(batch_size):
                if medical_lengths[i] == 0:
                    missing[i, 0] = 0

            # 检测文本模态是否有效
            if 'notes' in batch and batch['notes']['embeddings'] is not None:
                for i in range(batch_size):
                    if note_lengths[i] == 0:
                        missing[i, 1] = 0
            else:
                missing[:, 1] = 0

            # 检测lab模态是否有效
            if 'labs' in batch and batch['labs']['times'] is not None:
                for i in range(batch_size):
                    if lab_lengths[i] == 0:
                        missing[i, 2] = 0
            else:
                missing[:, 2] = 0

        # ===== 通过融合transformer处理 =====
        outputs, _ = self.fusion_transformer(
            enc_outputs=[medical_emb, note_emb, lab_emb],
            lengths=[medical_lengths, note_lengths, lab_lengths],
            fusion_idx=2,
            missing=missing  # 缺失模态信息
        )

        # 获取各模态的CLS token
        medical_cls = self.layer_norm(outputs[0][:, 0, :])  # 医疗代码的CLS token
        note_cls = self.layer_norm(outputs[1][:, 0, :])  # 文本注释的CLS token
        lab_cls = self.layer_norm(outputs[2][:, 0, :])  # 实验室检测的CLS token

        # 初始化融合后的特征
        fused_cls = torch.zeros_like(medical_cls)

        # 对每个样本进行模态融合
        for i in range(batch_size):
            # 获取有效模态的掩码
            valid_modalities = missing[i]  # [3]，值为0或1

            # 如果至少有一个模态存在
            if valid_modalities.sum() > 0:
                # 收集当前样本的所有有效模态的CLS token
                valid_cls_tokens = []
                if valid_modalities[0] == 1:
                    valid_cls_tokens.append(medical_cls[i:i + 1])
                if valid_modalities[1] == 1:
                    valid_cls_tokens.append(note_cls[i:i + 1])
                if valid_modalities[2] == 1:
                    valid_cls_tokens.append(lab_cls[i:i + 1])

                if len(valid_cls_tokens) == 1:
                    # 如果只有一个有效模态，直接使用
                    fused_cls[i] = valid_cls_tokens[0]
                else:
                    # 如果有多个有效模态，使用注意力机制融合
                    all_cls_tokens = []
                    if valid_modalities[0] == 1:
                        all_cls_tokens.append(medical_cls[i])
                    else:
                        all_cls_tokens.append(torch.zeros_like(medical_cls[i]))

                    if valid_modalities[1] == 1:
                        all_cls_tokens.append(note_cls[i])
                    else:
                        all_cls_tokens.append(torch.zeros_like(note_cls[i]))

                    if valid_modalities[2] == 1:
                        all_cls_tokens.append(lab_cls[i])
                    else:
                        all_cls_tokens.append(torch.zeros_like(lab_cls[i]))

                    # 将所有CLS token堆叠
                    stacked_cls = torch.stack(all_cls_tokens, dim=0)  # [3, d_model]

                    # 计算模态权重
                    weights = self.cls_fusion_attention(stacked_cls.mean(dim=0, keepdim=True))  # [1, 3]
                    # 应用掩码，将缺失模态的权重设为0
                    masked_weights = weights * valid_modalities.unsqueeze(0).float()
                    # 重新归一化权重
                    if masked_weights.sum() > 0:
                        normalized_weights = masked_weights / masked_weights.sum()
                    else:
                        normalized_weights = masked_weights

                    # 加权融合
                    weighted_sum = torch.zeros_like(medical_cls[i])
                    if valid_modalities[0] == 1:
                        weighted_sum += normalized_weights[0, 0] * medical_cls[i]
                    if valid_modalities[1] == 1:
                        weighted_sum += normalized_weights[0, 1] * note_cls[i]
                    if valid_modalities[2] == 1:
                        weighted_sum += normalized_weights[0, 2] * lab_cls[i]

                    fused_cls[i] = weighted_sum
            else:
                # 如果所有模态都缺失（极端情况），使用默认特征
                fused_cls[i] = torch.zeros_like(medical_cls[i])

        # 通过分类器
        logits = self.classifier(fused_cls)

        return logits


def handle_missing_modality(batch):
    """
    处理缺失模态，返回missing参数

    参数:
    batch: 批次数据

    返回:
    missing: 每个样本的模态存在标志 [batch_size, 3]
             第一列：医疗代码模态，1表示存在，0表示缺失
             第二列：文本模态，1表示存在，0表示缺失
             第三列：lab模态，1表示存在，0表示缺失
    """
    batch_size = batch['demographic'].size(0)
    missing = torch.ones(batch_size, 3, dtype=torch.long, device=batch['demographic'].device)

    # 检查医疗代码模态
    if 'medical' in batch and batch['medical']['codes'] is not None:
        for i in range(batch_size):
            has_valid_codes = batch['medical']['mask'][i].any()
            if not has_valid_codes:
                missing[i, 0] = 0
    else:
        missing[:, 0] = 0

    # 检查notes模态
    if 'notes' in batch and batch['notes']['embeddings'] is not None:
        for i in range(batch_size):
            has_valid_notes = batch['notes']['mask'][i].any()
            if not has_valid_notes:
                missing[i, 1] = 0
    else:
        missing[:, 1] = 0

    # 检查labs模态
    if 'labs' in batch and batch['labs']['times'] is not None:
        for i in range(batch_size):
            has_valid_labs = batch['labs']['mask'][i].any()
            if not has_valid_labs:
                missing[i, 2] = 0
    else:
        missing[:, 2] = 0

    return missing