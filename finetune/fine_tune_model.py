from transformer import *

class ReadmissionPredictor(nn.Module):
    """
    基于简化预训练EHR模型的再入院预测器
    只使用四个核心交互路径：诊断-出院摘要，出院摘要-诊断，药物-出院摘要，出院摘要-药物
    """

    def __init__(self, pretrained_model, hidden_size=256, dropout_rate=0.1):
        super(ReadmissionPredictor, self).__init__()
        self.ehr_model = pretrained_model

        # 冻结预训练模型的部分参数
        self._freeze_params()

        # 获取每个交互路径的表示维度
        fusion_dim = hidden_size * 4  # 四个交互路径

        # 添加用于再入院预测的分类头
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1),
        )

    def _freeze_params(self):
        """冻结预训练模型的部分参数"""
        # 冻结嵌入层
        for param in self.ehr_model.embeddings.parameters():
            param.requires_grad = False

        # 冻结编码器的前两层（保留对高层的微调）
        for modality, encoder in self.ehr_model.encoders.items():
            for i, layer in enumerate(encoder.layers):
                if i < 2:  # 只冻结前两层
                    for param in layer.parameters():
                        param.requires_grad = False

    def forward(self, batch):
        """
        前向传播，从简化的四个交互路径中提取再入院预测特征

        参数:
            batch: 包含多模态数据的批次

        返回:
            readmission_prob: 再入院概率
            enhanced_representations: 增强的表示（可用于解释性分析）
        """
        # 获取批次大小
        batch_size = batch['label'].size(0)
        device = batch['label'].device

        # 获取嵌入和掩码
        embeddings, masks = self.ehr_model.embeddings(batch)

        # 使用各自的编码器处理嵌入
        encoded_sequences = {}
        encoded_cls = {}

        for modality, embedding in embeddings.items():
            if modality in self.ehr_model.encoders:
                # 使用编码器获取序列表示和CLS表示
                seq, cls = self.ehr_model.encoders[modality](embedding, masks.get(modality, None))
                encoded_sequences[modality] = seq
                encoded_cls[modality] = cls

        # 执行跨模态注意力融合 - 只保留四个核心路径
        enhanced_representations = {}

        # 1. 诊断-出院摘要交互
        if 'diagnosis' in encoded_cls and 'discharge_summary' in encoded_sequences:
            dx_enhanced_by_ds = self.ehr_model.dx2dis(
                encoded_cls['diagnosis'].unsqueeze(1),
                encoded_sequences['discharge_summary'],
                masks.get('discharge_summary', None)
            ).squeeze(1)
            # 添加残差连接
            dx_enhanced_by_ds = dx_enhanced_by_ds + encoded_cls['diagnosis']
            dx_enhanced_by_ds = self.ehr_model.layer_norm(dx_enhanced_by_ds)
            enhanced_representations['dx_by_ds'] = dx_enhanced_by_ds

        # 2. 出院摘要-诊断交互
        if 'discharge_summary' in encoded_cls and 'diagnosis' in encoded_sequences:
            dis_enhanced_by_dx = self.ehr_model.dis2dx(
                encoded_cls['discharge_summary'].unsqueeze(1),
                encoded_sequences['diagnosis'],
                masks.get('diagnosis', None)
            ).squeeze(1)
            # 添加残差连接
            dis_enhanced_by_dx = dis_enhanced_by_dx + encoded_cls['discharge_summary']
            dis_enhanced_by_dx = self.ehr_model.layer_norm(dis_enhanced_by_dx)
            enhanced_representations['dis_by_dx'] = dis_enhanced_by_dx

        # 3. 药物-出院摘要交互
        if 'medication' in encoded_cls and 'discharge_summary' in encoded_sequences:
            med_enhanced_by_ds = self.ehr_model.med2dis(
                encoded_cls['medication'].unsqueeze(1),
                encoded_sequences['discharge_summary'],
                masks.get('discharge_summary', None)
            ).squeeze(1)
            # 添加残差连接
            med_enhanced_by_ds = med_enhanced_by_ds + encoded_cls['medication']
            med_enhanced_by_ds = self.ehr_model.layer_norm(med_enhanced_by_ds)
            enhanced_representations['med_by_ds'] = med_enhanced_by_ds

        # 4. 出院摘要-药物交互
        if 'discharge_summary' in encoded_cls and 'medication' in encoded_sequences:
            dis_enhanced_by_med = self.ehr_model.dis2med(
                encoded_cls['discharge_summary'].unsqueeze(1),
                encoded_sequences['medication'],
                masks.get('medication', None)
            ).squeeze(1)
            # 添加残差连接
            dis_enhanced_by_med = dis_enhanced_by_med + encoded_cls['discharge_summary']
            dis_enhanced_by_med = self.ehr_model.layer_norm(dis_enhanced_by_med)
            enhanced_representations['dis_by_med'] = dis_enhanced_by_med

        # 定义所需的表示键
        representation_keys = ['dx_by_ds', 'dis_by_dx', 'med_by_ds', 'dis_by_med']

        # 收集所有可用的表示
        fusion_features = []
        for key in representation_keys:
            if key in enhanced_representations:
                fusion_features.append(enhanced_representations[key])
            else:
                # 如果表示不可用，使用零向量填充
                hidden_size = next(iter(encoded_cls.values())).size(1)
                fusion_features.append(torch.zeros(batch_size, hidden_size, device=device))

        # 将所有特征连接起来
        fused_representation = torch.cat(fusion_features, dim=1)

        # 通过分类器预测再入院概率
        readmission_logit = self.classifier(fused_representation)
        readmission_prob = torch.sigmoid(readmission_logit)

        return readmission_prob.squeeze(-1), enhanced_representations