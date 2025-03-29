import torch
import torch.nn as nn
import torch.nn.functional as F
from disease_codes import DISEASE_CODES, disease_weights  # 假设disease_weights仍然需要


class LSTMModel(nn.Module):
    def __init__(self, num_codes, hidden_dim, disease_names, demographic_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_codes = num_codes
        self.hidden_dim = hidden_dim
        self.disease_names = disease_names
        self.disease_codes = DISEASE_CODES
        self.num_layers = num_layers

        # 基础组件
        self.code_transform = nn.Linear(num_codes, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 分类器 - 使用LSTM最后一个隐藏状态
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + demographic_dim, hidden_dim),  # *2 因为是双向LSTM
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_codes)  # 只预测疾病，不包括死亡
        )

        self.criterion = nn.BCELoss()

    def get_disease_probability(self, logits, disease_name):
        """获取疾病的预测概率"""
        disease_codes = self.disease_codes[disease_name]
        probs = torch.sigmoid(logits[torch.tensor([code - 1 for code in disease_codes], device=logits.device)])
        return torch.max(probs).view(1)

    def process_single_prediction(self, history_codes, demographic, device):
        """处理单个预测的历史记录"""
        visit_embeddings = []
        for visit in history_codes:
            if len(visit) == 0:
                continue
            one_hot = torch.zeros(self.num_codes, device=device)
            visit_indices = torch.tensor(visit, device=device)
            one_hot.scatter_(0, visit_indices, 1.0)
            visit_embedding = self.code_transform(one_hot)
            visit_embeddings.append(visit_embedding)

        if not visit_embeddings:
            return None

        visit_embeddings = torch.stack(visit_embeddings).unsqueeze(0)

        # LSTM处理 - 只使用最后一个隐藏状态
        _, (hidden, _) = self.lstm(visit_embeddings)

        # 拼接最后一层的前向和后向隐藏状态
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)

        # 结合人口统计特征并预测
        combined_features = torch.cat([last_hidden, demographic.unsqueeze(0)], dim=1)
        return self.classifier(combined_features).squeeze()

    def forward(self, batch, disease_weights=None, is_training=True):
        demographic = batch['demographic']
        disease_data = batch['disease_data']
        batch_size = demographic.size(0)
        device = demographic.device

        if not is_training:
            predictions = {}
            # 初始化预测结果字典，仅针对疾病
            for disease_name in self.disease_names:
                predictions[disease_name] = {'predictions': [], 'labels': []}

            # 对每个样本进行预测
            for b in range(batch_size):
                # 疾病预测
                for disease_name in self.disease_names:
                    if disease_name not in disease_data[b] or disease_data[b][disease_name]['label'] == -1:
                        continue

                    try:
                        logits = self.process_single_prediction(
                            disease_data[b][disease_name]['history_codes'],
                            demographic[b],
                            device
                        )

                        if logits is not None:
                            prob = self.get_disease_probability(logits, disease_name)
                            predictions[disease_name]['predictions'].append(prob)
                            predictions[disease_name]['labels'].append(disease_data[b][disease_name]['label'])
                    except Exception as e:
                        print(f"Error in validation for disease {disease_name}: {str(e)}")
                        continue

            return predictions

        else:  # 训练模式
            total_loss = torch.tensor([0.0], device=device)
            total_weighted_samples = 0

            for b in range(batch_size):
                # 疾病预测和损失计算
                for disease_name in self.disease_names:
                    if disease_name not in disease_data[b] or disease_data[b][disease_name]['label'] == -1:
                        continue

                    try:
                        logits = self.process_single_prediction(
                            disease_data[b][disease_name]['history_codes'],
                            demographic[b],
                            device
                        )

                        if logits is not None:
                            prob = self.get_disease_probability(logits, disease_name)
                            label = disease_data[b][disease_name]['label']
                            weight = disease_weights.get(disease_name, 1.0) if disease_weights else 1.0
                            loss = self.criterion(
                                prob,
                                torch.tensor([label], device=device, dtype=torch.float)
                            )
                            total_loss += loss * weight
                            total_weighted_samples += 1
                    except Exception as e:
                        print(f"Error in training for disease {disease_name}: {str(e)}")
                        continue

            return total_loss / total_weighted_samples if total_weighted_samples > 0 else torch.tensor(0.0,
                                                                                                       device=device)