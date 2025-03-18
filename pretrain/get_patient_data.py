import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random

class PatientDataset(Dataset):
    def __init__(self, data, mappings,code_mappings, index_set, do_mask=True, mask_prob=0.15):
        """
        初始化以visit为单位的病人数据集

        参数:
        data: 已加载的visit数据列表，每个元素是一个visit记录
        mappings: 特征映射字典，将分类变量映射到索引（用于demographic）
        code_mappings: 包含dx_code_mapping和rx_code_mapping的字典（用于医疗代码）
        index_set: 索引集合，用于过滤有效的代码
        do_mask: 是否执行掩码，默认为True（仅对代码生效）
        mask_prob: 掩码概率，默认为0.15
        """
        self.data = data
        self.mappings = mappings
        self.dx_code_mapping = code_mappings['dx_code_mapping']
        self.rx_code_mapping = code_mappings['rx_code_mapping']
        self.index_set = index_set
        self.do_mask = do_mask
        self.mask_prob = mask_prob
        self.dx_mask_token = len(self.dx_code_mapping)
        self.rx_mask_token = len(self.rx_code_mapping)
        self.note_mask_token = 100000  # 全局mask token，用于note相关的代码

        # 计算每个特征的最大索引（用于demographic）
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
        if demo and 'age' in demo:
            features.append(np.log1p(float(demo['age'])))
        else:
            features.append(np.log1p(0.0))
        for feature in ['gender', 'race', 'marital_status', 'language']:
            if demo and feature in demo:
                feature_value = demo[feature]
                if pd.isna(feature_value):
                    feature_value = 'nan'
            else:
                feature_value = 'nan'
            feat_idx = self.mappings[feature].get(feature_value, self.mappings[feature]['nan'])
            one_hot = [0] * (self.max_indices[feature] + 1)
            one_hot[feat_idx] = 1
            features.extend(one_hot)
        return torch.tensor(features, dtype=torch.float32)

    def deduplicate_codes(self, codes, times):
        """
        对代码去重，保留最早出现的时间

        参数:
        codes: 代码列表
        times: 对应的时间列表

        返回:
        dedup_codes: 去重后的代码列表
        dedup_times: 去重后的时间列表
        """
        if not codes:
            return [], []
        code_time_dict = {}
        for code, time in zip(codes, times):
            if code not in code_time_dict or time < code_time_dict[code]:
                code_time_dict[code] = time
        dedup_codes = list(code_time_dict.keys())
        dedup_times = [code_time_dict[code] for code in dedup_codes]
        return dedup_codes, dedup_times

    def __getitem__(self, idx):
        visit_data = self.data[idx]
        demographic_features = self.process_demographic_features(visit_data['demographics'])
        visit = visit_data['visit']
        label = visit['30_days_readmission']

        # 为三种医疗代码类型分别创建事件列表
        diagnosis_events = []
        medication_events = []
        drg_events = []
        for event_type in ['ccs_events', 'icd10_events', 'icd9_events', 'phecode_events']:
            if event_type in visit:
                events = [
                    {'code_index': self.dx_code_mapping.get(event['code_index'], 0), 'relative_time': event['relative_time']}
                    for event in visit[event_type]
                    if 'code_index' in event and
                       (not self.index_set or event['code_index'] in self.index_set) and
                       not pd.isna(event['relative_time'])
                ]
                diagnosis_events.extend(events)
        if 'rxnorm_events' in visit:
            medication_events = [
                {'code_index': self.rx_code_mapping.get(event['code_index'], 0), 'relative_time': event['relative_time']}
                for event in visit['rxnorm_events']
                if 'code_index' in event and
                   (not self.index_set or event['code_index'] in self.index_set) and
                   not pd.isna(event['relative_time'])
            ]
        for event_type in ['drg_APR_events', 'drg_HCFA_events']:
            if event_type in visit:
                events = [
                    {'code_index': event['code_index'], 'relative_time': event['relative_time']}
                    for event in visit[event_type]
                    if 'code_index' in event and
                       (not self.index_set or event['code_index'] in self.index_set) and
                       not pd.isna(event['relative_time'])
                ]
                drg_events.extend(events)

        # 按时间排序
        diagnosis_events.sort(key=lambda x: x['relative_time'])
        medication_events.sort(key=lambda x: x['relative_time'])
        drg_events.sort(key=lambda x: x['relative_time'])

        # 提取代码和时间，然后去重
        diagnosis_codes = [e['code_index'] for e in diagnosis_events]
        diagnosis_times = [e['relative_time'] for e in diagnosis_events]
        diagnosis_codes, diagnosis_times = self.deduplicate_codes(diagnosis_codes, diagnosis_times)

        medication_codes = [e['code_index'] for e in medication_events]
        medication_times = [e['relative_time'] for e in medication_events]
        medication_codes, medication_times = self.deduplicate_codes(medication_codes, medication_times)

        drg_codes = [e['code_index'] for e in drg_events]
        drg_times = [e['relative_time'] for e in drg_events]
        drg_codes, drg_times = self.deduplicate_codes(drg_codes, drg_times)

        # 处理文本相关数据（不去重）
        dis_embeddings = []
        dis_times = []
        rad_embeddings = []
        rad_times = []
        dis_codes = []
        dis_codes_times = []
        rad_codes = []
        rad_codes_times = []

        if 'dis_embeddings' in visit and visit['dis_embeddings']:
            for emb in visit['dis_embeddings']:
                if 'embedding' in emb and 'relative_time' in emb and not pd.isna(emb['relative_time']):
                    dis_embeddings.append(emb['embedding'])
                    dis_times.append(emb['relative_time'])
        if 'rad_embeddings' in visit and visit['rad_embeddings']:
            for emb in visit['rad_embeddings']:
                if 'embedding' in emb and 'relative_time' in emb and not pd.isna(emb['relative_time']):
                    rad_embeddings.append(emb['embedding'])
                    rad_times.append(emb['relative_time'])
        if 'dis_codes' in visit and visit['dis_codes']:
            for code in visit['dis_codes']:
                if 'code_index' in code and 'relative_time' in code:
                    dis_codes.append(code['code_index'])
                    dis_codes_times.append(code['relative_time'])
            dis_codes, dis_codes_times = self.deduplicate_codes(dis_codes, dis_codes_times)
        if 'rad_codes' in visit and visit['rad_codes']:
            for code in visit['rad_codes']:
                if 'code_index' in code and 'relative_time' in code:
                    rad_codes.append(code['code_index'])
                    rad_codes_times.append(code['relative_time'])
            rad_codes, rad_codes_times = self.deduplicate_codes(rad_codes, rad_codes_times)

        # 处理lab事件
        lab_events = []
        if 'lab_events' in visit:
            events = [
                {'code_index': event['code_index'], 'relative_time': event['relative_time'], 'value': event['standardized_value']}
                for event in visit['lab_events']
                if 'code_index' in event and 'relative_time' in event and 'standardized_value' in event and not pd.isna(event['relative_time'])
            ]
            lab_events.extend(events)
        lab_events.sort(key=lambda x: x['relative_time'])
        lab_codes = [e['code_index'] for e in lab_events]
        lab_times = [e['relative_time'] for e in lab_events]
        lab_values = [e['value'] for e in lab_events]
        lab_codes, lab_times = self.deduplicate_codes(lab_codes, lab_times)
        lab_values = [lab_values[i] for i in range(len(lab_times))]  # 保持值与去重后的时间对应

        # 保存原始代码（去重后的版本）
        original_dx_codes = list(diagnosis_codes)
        original_med_codes = list(medication_codes)
        original_drg_codes = list(drg_codes)
        original_dis_codes = list(dis_codes)
        original_rad_codes = list(rad_codes)
        original_lab_codes = list(lab_codes)

        # 应用掩码（如果启用，区分诊断/药物和note相关代码）
        if self.do_mask:
            for i in range(len(diagnosis_codes)):
                if random.random() < self.mask_prob:
                    diagnosis_codes[i] = self.dx_mask_token
            for i in range(len(medication_codes)):
                if random.random() < self.mask_prob:
                    medication_codes[i] = self.rx_mask_token
            for i in range(len(drg_codes)):
                if random.random() < self.mask_prob:
                    drg_codes[i] = self.note_mask_token
            for i in range(len(dis_codes)):
                if random.random() < self.mask_prob:
                    dis_codes[i] = self.note_mask_token
            for i in range(len(rad_codes)):
                if random.random() < self.mask_prob:
                    rad_codes[i] = self.note_mask_token
            for i in range(len(lab_codes)):
                if random.random() < self.mask_prob:
                    lab_codes[i] = self.note_mask_token

        # 构建返回的数据结构
        return {
            'demographic': demographic_features,
            'diagnosis': {'codes': diagnosis_codes, 'times': diagnosis_times},
            'medication': {'codes': medication_codes, 'times': medication_times},
            'drg': {'codes': drg_codes, 'times': drg_times},
            'discharge_summary': {'embeddings': dis_embeddings, 'times': dis_times},
            'discharge_codes': {'codes': dis_codes, 'times': dis_codes_times},
            'radiology_report': {'embeddings': rad_embeddings, 'times': rad_times},
            'radiology_codes': {'codes': rad_codes, 'times': rad_codes_times},
            'lab': {'codes': lab_codes, 'times': lab_times, 'values': lab_values},
            'original_codes': {
                'diagnosis': original_dx_codes,
                'medication': original_med_codes,
                'drg': original_drg_codes,
                'discharge_codes': original_dis_codes,
                'radiology_codes': original_rad_codes,
                'lab': original_lab_codes
            },
            'label': label,
            'patient_id': visit_data['patient_id'],
            'visit_idx': visit_data['visit_idx']
        }

def custom_collate(batch, fixed_lengths=None):
    """
    自定义collate函数，对不同长度的序列进行填充
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

    # 处理原始代码 (original_codes)
    if 'original_codes' in batch[0]:
        result['original_codes'] = {}

        # 定义需要处理的原始代码模态及其对应的固定长度key
        orig_code_modalities = [
            ('diagnosis', 'diagnosis_events'),
            ('medication', 'medication_events'),
            ('drg', 'drg_events'),
            ('discharge_codes', 'discharge_codes_events'),
            ('radiology_codes', 'radiology_codes_events'),
            ('lab', 'lab_events')
        ]

        # 处理所有原始代码模态
        for modal, length_key in orig_code_modalities:
            if modal in batch[0]['original_codes']:
                max_len = fixed_lengths[length_key]
                orig_codes_batch = []

                for item in batch:
                    orig_codes = item['original_codes'][modal]
                    orig_len = len(orig_codes)

                    # 截断或填充
                    if orig_len > max_len:
                        orig_codes = orig_codes[-max_len:]
                    else:
                        padding_size = max_len - orig_len
                        orig_codes = orig_codes + [0] * padding_size

                    orig_codes_batch.append(torch.tensor(orig_codes, dtype=torch.long))

                result['original_codes'][modal] = torch.stack(orig_codes_batch)

    return result