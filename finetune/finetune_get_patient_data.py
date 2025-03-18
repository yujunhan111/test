import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random
import copy


class PatientDatasetForFinetune(Dataset):
    def __init__(self, data, mappings, index_set):
        """
        初始化用于微调任务的病人数据集
        与预训练不同，微调数据集不应用掩码，直接使用完整数据

        参数:
        data: 已加载的病人数据列表
        mappings: 特征映射字典，将分类变量映射到索引
        index_set: 索引集合，用于过滤有效的代码
        """
        self.data = data
        self.mappings = mappings
        self.index_set = index_set

        # 计算每个特征的最大索引
        self.max_indices = {
            'gender': max(v for v in mappings['gender'].values() if isinstance(v, (int, float))),
            'race': max(v for v in mappings['race'].values() if isinstance(v, (int, float))),
            'marital_status': max(v for v in mappings['marital_status'].values() if isinstance(v, (int, float))),
            'language': max(v for v in mappings['language'].values() if isinstance(v, (int, float)))
        }

    def __len__(self):
        return len(self.data)

    def process_demographic_features(self, demo):
        features = [1.0]

        # 处理age字段
        if demo and 'age' in demo:
            features.append(np.log1p(float(demo['age'])))
        else:
            # 如果demographics为空或没有age字段，使用默认值
            features.append(np.log1p(0.0))  # 使用0作为缺失值

        # 处理其他demographic特征
        for feature in ['gender', 'race', 'marital_status', 'language']:
            if demo and feature in demo:
                feature_value = demo[feature]
                if pd.isna(feature_value):
                    feature_value = 'nan'
            else:
                # 特征不存在，使用'nan'
                feature_value = 'nan'

            # 获取特征索引，使用'nan'作为默认值
            feat_idx = self.mappings[feature].get(feature_value, self.mappings[feature]['nan'])
            one_hot = [0] * (self.max_indices[feature] + 1)
            one_hot[feat_idx] = 1
            features.extend(one_hot)

        # 转换为tensor
        demographic_features = torch.tensor(features, dtype=torch.float32)

        return demographic_features

    def __getitem__(self, idx):
        """获取单个病人的数据，保留时间信息，为微调任务准备"""
        patient_data = self.data[idx]

        # 处理人口统计学特征
        demographic_features = self.process_demographic_features(patient_data['demographics'])


        visits = patient_data['visits']

        # 获取标签(readmission)
        label = visits[-2]['30_days_readmission']
        # 为三种医疗代码类型分别创建事件列表
        diagnosis_events = []
        medication_events = []
        drg_events = []

        # 收集诊断事件
        for visit in visits[:-1]:  # 除了最后一次visit
            for event_type in ['ccs_events', 'icd10_events', 'icd9_events', 'phecode_events']:
                if event_type in visit:
                    events = [
                        {
                            'code_index': event['code_index'],
                            'relative_time': event['relative_time']
                        }
                        for event in visit[event_type]
                        if 'code_index' in event and
                           (not self.index_set or event['code_index'] in self.index_set) and
                           not pd.isna(event['relative_time'])
                    ]
                    diagnosis_events.extend(events)

        # 收集药物事件
        for visit in visits[:-1]:
            if 'rxnorm_events' in visit:
                events = [
                    {
                        'code_index': event['code_index'],
                        'relative_time': event['relative_time']
                    }
                    for event in visit['rxnorm_events']
                    if 'code_index' in event and
                       (not self.index_set or event['code_index'] in self.index_set) and
                       not pd.isna(event['relative_time'])
                ]
                medication_events.extend(events)

        # 收集DRG事件
        for visit in visits[:-1]:
            for event_type in ['drg_APR_events', 'drg_HCFA_events']:
                if event_type in visit:
                    events = [
                        {
                            'code_index': event['code_index'],
                            'relative_time': event['relative_time']
                        }
                        for event in visit[event_type]
                        if 'code_index' in event and
                           (not self.index_set or event['code_index'] in self.index_set) and
                           not pd.isna(event['relative_time'])
                    ]
                    drg_events.extend(events)

        # 对三种医疗代码事件分别按时间排序
        diagnosis_events.sort(key=lambda x: x['relative_time'])
        medication_events.sort(key=lambda x: x['relative_time'])
        drg_events.sort(key=lambda x: x['relative_time'])

        # 提取诊断代码和时间
        diagnosis_codes = [e['code_index'] for e in diagnosis_events]
        diagnosis_times = [e['relative_time'] for e in diagnosis_events]

        # 提取药物代码和时间
        medication_codes = [e['code_index'] for e in medication_events]
        medication_times = [e['relative_time'] for e in medication_events]

        # 提取DRG代码和时间
        drg_codes = [e['code_index'] for e in drg_events]
        drg_times = [e['relative_time'] for e in drg_events]

        # 分别处理四种文本相关数据
        dis_embeddings = []
        dis_times = []
        rad_embeddings = []
        rad_times = []
        dis_codes = []
        dis_codes_times = []
        rad_codes = []
        rad_codes_times = []

        # 收集出院摘要嵌入
        for visit in visits[:-1]:
            if 'dis_embeddings' in visit and visit['dis_embeddings']:
                for emb in visit['dis_embeddings']:
                    if 'embedding' in emb and 'relative_time' in emb and not pd.isna(emb['relative_time']):
                        dis_embeddings.append(emb['embedding'])
                        dis_times.append(emb['relative_time'])

        # 收集放射学报告嵌入
        for visit in visits[:-1]:
            if 'rad_embeddings' in visit and visit['rad_embeddings']:
                for emb in visit['rad_embeddings']:
                    if 'embedding' in emb and 'relative_time' in emb and not pd.isna(emb['relative_time']):
                        rad_embeddings.append(emb['embedding'])
                        rad_times.append(emb['relative_time'])

        # 收集出院摘要代码
        for visit in visits[:-1]:
            if 'dis_codes' in visit and visit['dis_codes']:
                for code in visit['dis_codes']:
                    if 'code_index' in code and 'relative_time' in code:
                        dis_codes.append(code['code_index'])
                        dis_codes_times.append(code['relative_time'])

        # 收集放射学报告代码
        for visit in visits[:-1]:
            if 'rad_codes' in visit and visit['rad_codes']:
                for code in visit['rad_codes']:
                    if 'code_index' in code and 'relative_time' in code:
                        rad_codes.append(code['code_index'])
                        rad_codes_times.append(code['relative_time'])

        # 处理lab事件
        lab_events = []
        for visit in visits[:-1]:
            if 'lab_events' in visit:
                events = [
                    {
                        'code_index': event['code_index'],
                        'relative_time': event['relative_time'],
                        'value': event['standardized_value']
                    }
                    for event in visit['lab_events']
                    if 'code_index' in event and
                       'relative_time' in event and
                       'standardized_value' in event and
                       not pd.isna(event['relative_time'])
                ]
                lab_events.extend(events)

        # 按时间排序lab事件
        lab_events.sort(key=lambda x: x['relative_time'])

        # 提取lab代码、时间和值
        lab_codes = [e['code_index'] for e in lab_events]
        lab_times = [e['relative_time'] for e in lab_events]
        lab_values = [e['value'] for e in lab_events]

        # 构建返回的数据结构
        return {
            'demographic': demographic_features,
            'diagnosis': {
                'codes': diagnosis_codes,
                'times': diagnosis_times
            },
            'medication': {
                'codes': medication_codes,
                'times': medication_times
            },
            'drg': {
                'codes': drg_codes,
                'times': drg_times
            },
            'discharge_summary': {
                'embeddings': dis_embeddings,
                'times': dis_times
            },
            'discharge_codes': {
                'codes': dis_codes,
                'times': dis_codes_times
            },
            'radiology_report': {
                'embeddings': rad_embeddings,
                'times': rad_times
            },
            'radiology_codes': {
                'codes': rad_codes,
                'times': rad_codes_times
            },
            'lab': {
                'codes': lab_codes,
                'times': lab_times,
                'values': lab_values
            },
            'label': label,
            'patient_id': patient_data['patient_id']
        }


def custom_collate_finetune(batch, fixed_lengths=None):
    """
    用于微调任务的自定义collate函数，对不同长度的序列进行填充
    使用循环处理相似结构的模态数据，更加简洁

    参数:
    batch: 一批次的病人数据
    fixed_lengths: 指定各类事件的固定长度
    """
    if fixed_lengths is None:
        fixed_lengths = {
            'diagnosis_events': 100,
            'medication_events': 100,
            'drg_events': 30,
            'discharge_summary_events': 10,
            'discharge_codes_events': 200,
            'radiology_report_events': 10,
            'radiology_codes_events': 200,
            'lab_events': 200
        }

    batch_size = len(batch)
    result = {}

    # 处理demographic数据
    if 'demographic' in batch[0]:
        result['demographic'] = torch.stack([item['demographic'] for item in batch])

    # 处理标签
    if 'label' in batch[0]:
        result['label'] = torch.tensor([item['label'] for item in batch], dtype=torch.long)

    # 处理patient_id
    if 'patient_id' in batch[0]:
        result['patient_id'] = [item['patient_id'] for item in batch]

    # 定义需要处理的代码模态及其对应的固定长度key
    code_modalities = [
        ('diagnosis', 'diagnosis_events'),
        ('medication', 'medication_events'),
        ('drg', 'drg_events'),
        ('discharge_codes', 'discharge_codes_events'),
        ('radiology_codes', 'radiology_codes_events')
    ]

    # 处理所有代码模态
    for modal, length_key in code_modalities:
        if modal in batch[0]:
            max_len = fixed_lengths[length_key]
            codes_batch = []
            times_batch = []
            masks_batch = []
            lengths_batch = []

            for item in batch:
                codes = item[modal]['codes']
                times = item[modal]['times']
                orig_len = len(codes)

                # 记录实际长度
                lengths_batch.append(orig_len)

                # 截断或填充
                if orig_len > max_len:
                    # 保留最新的事件
                    codes = codes[-max_len:]
                    times = times[-max_len:]
                    mask = [1] * max_len
                else:
                    # 需要填充
                    padding_size = max_len - orig_len
                    mask = [1] * orig_len + [0] * padding_size
                    codes = codes + [0] * padding_size
                    times = times + [0] * padding_size

                # 转换为张量
                codes_batch.append(torch.tensor(codes, dtype=torch.long))
                times_batch.append(torch.tensor(times, dtype=torch.float))
                masks_batch.append(torch.tensor(mask, dtype=torch.bool))

            # 合并批次
            result[modal] = {
                'codes': torch.stack(codes_batch),
                'times': torch.stack(times_batch),
                'mask': torch.stack(masks_batch),
                'lengths': torch.tensor(lengths_batch, dtype=torch.long)
            }

    # 处理实验室检测（特殊情况，因为有values）
    if 'lab' in batch[0]:
        max_len = fixed_lengths['lab_events']
        codes_batch = []
        times_batch = []
        values_batch = []
        masks_batch = []
        lengths_batch = []

        for item in batch:
            codes = item['lab']['codes']
            times = item['lab']['times']
            values = item['lab']['values']
            orig_len = len(times)

            # 记录实际长度
            lengths_batch.append(orig_len)

            # 截断或填充
            if orig_len > max_len:
                # 保留最新的事件
                codes = codes[-max_len:]
                times = times[-max_len:]
                values = values[-max_len:]
                mask = [1] * max_len
            else:
                # 需要填充
                padding_size = max_len - orig_len
                mask = [1] * orig_len + [0] * padding_size
                codes = codes + [0] * padding_size
                times = times + [0] * padding_size
                values = values + [0] * padding_size

            # 转换为张量
            codes_batch.append(torch.tensor(codes, dtype=torch.long))
            times_batch.append(torch.tensor(times, dtype=torch.float))
            values_batch.append(torch.tensor(values, dtype=torch.float))
            masks_batch.append(torch.tensor(mask, dtype=torch.bool))

        # 合并批次
        result['lab'] = {
            'codes': torch.stack(codes_batch),
            'times': torch.stack(times_batch),
            'values': torch.stack(values_batch),
            'mask': torch.stack(masks_batch),
            'lengths': torch.tensor(lengths_batch, dtype=torch.long)
        }

    # 定义需要处理的嵌入模态
    embedding_modalities = [
        ('discharge_summary', 'discharge_summary_events'),
        ('radiology_report', 'radiology_report_events')
    ]

    # 处理所有嵌入模态
    for modal, length_key in embedding_modalities:
        if modal in batch[0]:
            max_len = fixed_lengths[length_key]
            emb_batch = []
            times_batch = []
            masks_batch = []
            lengths_batch = []

            # 获取嵌入维度
            emb_dim = None
            for item in batch:
                if item[modal]['embeddings'] and len(item[modal]['embeddings']) > 0:
                    emb_dim = len(item[modal]['embeddings'][0])
                    break

            # 如果没有找到嵌入维度，跳过处理
            if emb_dim is not None:
                for item in batch:
                    embeddings = item[modal]['embeddings']
                    times = item[modal]['times']
                    orig_len = len(embeddings)

                    # 记录实际长度
                    lengths_batch.append(orig_len)

                    # 截断或填充
                    if orig_len > max_len:
                        # 保留最新的嵌入
                        embeddings = embeddings[-max_len:]
                        times = times[-max_len:]
                        mask = [1] * max_len
                    else:
                        # 需要填充
                        padding_size = max_len - orig_len
                        mask = [1] * orig_len + [0] * padding_size
                        # 创建零填充的嵌入
                        zero_padding = [np.zeros(emb_dim) for _ in range(padding_size)]
                        embeddings = embeddings + zero_padding
                        times = times + [0] * padding_size

                    # 转换为张量
                    emb_tensor = torch.tensor(np.array(embeddings), dtype=torch.float)
                    times_tensor = torch.tensor(times, dtype=torch.float)
                    mask_tensor = torch.tensor(mask, dtype=torch.bool)

                    emb_batch.append(emb_tensor)
                    times_batch.append(times_tensor)
                    masks_batch.append(mask_tensor)

                # 合并批次
                result[modal] = {
                    'embeddings': torch.stack(emb_batch) if emb_batch else None,
                    'times': torch.stack(times_batch) if times_batch else None,
                    'mask': torch.stack(masks_batch) if masks_batch else None,
                    'lengths': torch.tensor(lengths_batch, dtype=torch.long) if lengths_batch else None
                }

    return result

