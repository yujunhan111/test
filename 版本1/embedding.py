import pickle
import warnings
import numpy as np
import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime

# 导入自定义模块
from mappings import create_all_mappings
from get_patient_data import PatientDataset, custom_collate
from model import EHR_MBT_Model, handle_missing_modality
from transformer import *

# 忽略警告
warnings.filterwarnings("ignore")


class Args:
    def __init__(self, batch_size, code_vocab_size=12232):
        self.batch_size = batch_size
        self.code_vocab_size = code_vocab_size  # 医疗代码词汇表大小
        self.transformer_num_layers = 4  # Transformer层数
        self.transformer_num_head = 8  # 注意力头数
        self.transformer_dim = 256  # 模型维度
        self.dropout = 0.1  # Dropout比率
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fusion_startIdx = 2  # 从第几层开始进行模态融合


def load_model(model_path, batch_size):
    """加载预训练模型"""
    args = Args(batch_size=batch_size, code_vocab_size=12233)
    model = EHR_MBT_Model(args)

    checkpoint = torch.load(model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()

    return model, args


def extract_embeddings(model, args, test_data_path, code_dict, mappings):
    """从指定数据文件中提取患者嵌入"""
    # 获取医疗代码索引集合
    index_set = set(code_dict["index"])

    # 用于保存患者ID和其对应的嵌入
    patient_embeddings = {}

    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
        dataset = PatientDataset(test_data, mappings, index_set)
        test_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=custom_collate
        )

        # 测试循环
        for batch_idx, batch in enumerate(test_loader):
            # 将所有张量移到设备上
            for key in batch:
                if isinstance(batch[key], dict):
                    for subkey in batch[key]:
                        if isinstance(batch[key][subkey], torch.Tensor):
                            batch[key][subkey] = batch[key][subkey].to(args.device)
                elif isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(args.device)

            # 处理可能的缺失模态
            missing = handle_missing_modality(batch)

            # 获取患者ID
            patient_ids = batch['patient_id']

            # 使用修改后的前向传播，获取嵌入
            with torch.no_grad():
                # 获取批次大小
                batch_size = batch['demographic'].size(0)

                # 处理人口统计学特征
                demo_emb = model.demo_embedding(batch['demographic'])

                # ===== 处理医疗代码数据 =====
                medical_codes = batch['medical']['codes']
                medical_times = batch['medical']['times'].unsqueeze(-1)
                medical_types = batch['medical']['types']
                medical_mask = batch['medical']['mask']
                medical_lengths = batch['medical']['lengths']

                # 获取各种嵌入
                code_emb = model.code_embedding(medical_codes)
                time_emb = model.time_embedding(medical_times)
                type_emb = model.type_embedding(medical_types)

                # 合并嵌入
                medical_emb = code_emb + time_emb + type_emb

                # ===== 处理临床文本数据 =====
                if 'notes' in batch and batch['notes']['embeddings'] is not None:
                    note_emb = batch['notes']['embeddings']
                    note_times = batch['notes']['times'].unsqueeze(-1)
                    note_types = batch['notes']['types']
                    note_mask = batch['notes']['mask']
                    note_lengths = batch['notes']['lengths']

                    # 转换嵌入维度并添加时间和类型信息
                    note_value_emb = model.note_transform(note_emb)
                    note_time_emb = model.time_embedding(note_times)
                    note_type_emb = model.type_embedding(note_types)

                    # 合并嵌入
                    note_emb = note_value_emb + note_time_emb + note_type_emb
                else:
                    # 创建空的note嵌入和长度
                    note_emb = torch.zeros(batch_size, 1, model.model_dim, device=args.device)
                    note_lengths = torch.zeros(batch_size, dtype=torch.long, device=args.device)

                # ===== 处理实验室检测数据 =====
                if 'labs' in batch and batch['labs']['types'] is not None:
                    lab_times = batch['labs']['times'].unsqueeze(-1)
                    lab_values = batch['labs']['values'].unsqueeze(-1)
                    lab_types = batch['labs']['types']
                    lab_mask = batch['labs']['mask']
                    lab_lengths = batch['labs']['lengths']

                    # 获取各种嵌入
                    lab_time_emb = model.time_embedding(lab_times)
                    lab_value_emb = model.lab_value_transform(lab_values)
                    lab_type_emb = model.type_embedding(lab_types)

                    # 合并嵌入
                    lab_emb = lab_time_emb + lab_value_emb + lab_type_emb
                else:
                    # 创建空的lab嵌入和长度
                    lab_emb = torch.zeros(batch_size, 1, model.model_dim, device=args.device)
                    lab_lengths = torch.zeros(batch_size, dtype=torch.long, device=args.device)

                # 通过融合transformer处理
                outputs, _ = model.fusion_transformer(
                    enc_outputs=[medical_emb, note_emb, lab_emb],
                    lengths=[medical_lengths, note_lengths, lab_lengths],
                    fusion_idx=2,
                    missing=missing
                )

                # 获取各模态的CLS token
                medical_cls = model.layer_norm(outputs[0][:, 0, :])
                note_cls = model.layer_norm(outputs[1][:, 0, :])
                lab_cls = model.layer_norm(outputs[2][:, 0, :])

                # 初始化融合后的特征
                fused_cls = torch.zeros_like(medical_cls)

                # 对每个样本进行模态融合
                for i in range(batch_size):
                    # 获取有效模态的掩码
                    valid_modalities = missing[i]

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
                            stacked_cls = torch.stack(all_cls_tokens, dim=0)

                            # 计算模态权重
                            weights = model.cls_fusion_attention(stacked_cls.mean(dim=0, keepdim=True))
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
                        # 如果所有模态都缺失，使用默认特征
                        fused_cls[i] = torch.zeros_like(medical_cls[i])

                # 预测
                logits = model.classifier(fused_cls)
                probs = torch.sigmoid(logits).cpu().numpy()

                # 保存患者嵌入
                for i, patient_id in enumerate(patient_ids):
                    patient_embeddings[patient_id] = {
                        'embedding': fused_cls[i].cpu().numpy(),
                        'prediction': probs[i][0],
                        'label': batch['label'][i].item() if 'label' in batch else None
                    }

    return patient_embeddings


def save_patient_embeddings(patient_embeddings, output_path):
    """保存患者嵌入到文件"""
    data_to_save = {}
    for patient_id, data in patient_embeddings.items():
        data_to_save[patient_id] = {
            'embedding': data['embedding'],
            'prediction': data['prediction'],
            'label': data['label']
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data_to_save, f)


def main():
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 配置参数
    batch_size = 32

    # 获取模型路径
    model_path = r"C:\Users\yujun\PycharmProjects\ehr_v2\EHR_v2_初步尝试\复现medical_tri\第三个版本，三模态\experiments\experiment_20250305_080722\model_epoch11_valauc0.6850_20250305_114210.pth"

    # 获取测试数据路径
    base_path = r"C:\Users\yujun\Desktop\测验"
    test_data_paths = [
        os.path.join(base_path, "label_0_patients.pkl"),
        os.path.join(base_path, "label_1_patients.pkl"),
        os.path.join(base_path, "label_0_no_lab.pkl"),
        os.path.join(base_path, "label_1_no_lab.pkl"),
        os.path.join(base_path, "label_0_no_note.pkl"),
        os.path.join(base_path, "label_1_no_note.pkl"),
        os.path.join(base_path, "label_0_only_code.pkl"),
        os.path.join(base_path, "label_1_only_code.pkl")
    ]

    # 对应的输出文件名
    output_filenames = [
        "label_0_patients_embeddings.pkl",
        "label_1_patients_embeddings.pkl",
        "label_0_no_lab_embeddings.pkl",
        "label_1_no_lab_embeddings.pkl",
        "label_0_no_note_embeddings.pkl",
        "label_1_no_note_embeddings.pkl",
        "label_0_only_code_embeddings.pkl",
        "label_1_only_code_embeddings.pkl"
    ]

    # 获取患者字典和词嵌入数据路径
    patient_dict_path = r"C:\Users\yujun\Desktop\NUS学习材料\Nus 科研 EHR\ehr科研告一段落\patien_records_(note_hosp_icu_ed)/patients_dict.csv"
    code_dict_path = r"C:\Users\yujun\Desktop\DATA_汇总\dict_note_code.parquet"

    # 创建结果目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("./embeddings_results")
    output_dir.mkdir(exist_ok=True)
    result_dir = output_dir / f"embeddings_{timestamp}"
    result_dir.mkdir(exist_ok=True)

    # 加载患者字典和创建映射
    patient = pd.read_csv(patient_dict_path)
    mappings = create_all_mappings(patient)

    # 加载医疗代码字典
    code_dict = pd.read_parquet(code_dict_path)

    # 加载模型
    model, args = load_model(model_path, batch_size)

    # 依次处理每个文件并保存嵌入
    for test_data_path, output_filename in zip(test_data_paths, output_filenames):
        # 提取嵌入
        patient_embeddings = extract_embeddings(model, args, test_data_path, code_dict, mappings)

        # 设置保存路径
        output_path = result_dir / output_filename

        # 保存嵌入
        save_patient_embeddings(patient_embeddings, str(output_path))


if __name__ == "__main__":
    main()