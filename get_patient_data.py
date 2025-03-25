import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


class PatientDataset(Dataset):
    def __init__(self, data, mappings, index_set):
        """
        初始化病人数据集

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
            features.append(np.log1p(0.0))  # 使用0作为缺失值，或者您可以选择其他默认值

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
        """获取单个病人的数据，保留时间信息"""
        patient_data = self.data[idx]
        # 处理人口统计学特征
        demographic_features = self.process_demographic_features(patient_data['demographics'])
        # 提取倒数第二次及之前的visit
        visits = patient_data['visits']
        target_visits = visits[:-1]

        # 获取标签(倒数第二次visit的readmission)
        label = target_visits[-1]['30_days_readmission']

        # 为不同类型的医疗代码定义单独的事件列表
        event_groups = {
            'diagnosis': {
                'ccs_events': [],
                'icd10_events': [],
                'icd9_events': [],
                'phecode_events': []
            },
            'medication': {
                'rxnorm_events': []
            },
            'drg': {
                'drg_APR_events': [],
                'drg_HCFA_events': []
            }
        }

        # 收集各组的事件
        for visit in target_visits:
            for group_name, group_events in event_groups.items():
                for event_type in group_events.keys():
                    if event_type in visit:
                        events = [
                            {
                                'code_index': event['code_index'],
                                'relative_time': event['relative_time'],
                                'type': group_name  # 添加类型标记
                            }
                            for event in visit[event_type]
                            if 'code_index' in event and
                               (not self.index_set or event['code_index'] in self.index_set) and
                               not pd.isna(event['relative_time'])
                        ]
                        group_events[event_type].extend(events)

        # 合并所有医疗代码事件
        all_medical_events = []
        for group_name, group_events in event_groups.items():
            for event_list in group_events.values():
                all_medical_events.extend(event_list)

        # 按时间排序
        all_medical_events.sort(key=lambda x: x['relative_time'])

        # 提取代码和时间
        if all_medical_events:
            medical_codes = [e['code_index'] for e in all_medical_events]
            medical_times = [e['relative_time'] for e in all_medical_events]
            medical_types = [e['type'] for e in all_medical_events]
        else:
            medical_codes = []
            medical_times = []
            medical_types = []

        # 处理note嵌入，合并所有类型的嵌入
        all_note_embeddings = []
        note_types = []

        # 收集出院摘要嵌入
        for visit in target_visits:
            if 'dis_embeddings' in visit and visit['dis_embeddings']:
                for emb in visit['dis_embeddings']:
                    all_note_embeddings.append({
                        'embedding': emb['embedding'],
                        'relative_time': emb['relative_time'],
                        'type': 'discharge'
                    })

        # 收集放射学报告嵌入
        for visit in target_visits:
            if 'rad_embeddings' in visit and visit['rad_embeddings']:
                for emb in visit['rad_embeddings']:
                    all_note_embeddings.append({
                        'embedding': emb['embedding'],
                        'relative_time': emb['relative_time'],
                        'type': 'radiology'
                    })

        # 按时间排序所有文本嵌入
        all_note_embeddings.sort(key=lambda x: x['relative_time'])

        # 提取嵌入、时间和类型
        if all_note_embeddings:
            note_embeddings = [e['embedding'] for e in all_note_embeddings]
            note_times = [e['relative_time'] for e in all_note_embeddings]
            note_types = [e['type'] for e in all_note_embeddings]
        else:
            note_embeddings = []
            note_times = []
            note_types = []

        # 处理lab事件 - 新增部分
        all_lab_events = []

        # 收集lab事件
        for visit in target_visits:
            if 'lab_events' in visit and visit['lab_events']:
                for lab in visit['lab_events']:
                    if ('code_index' in lab and
                            'relative_time' in lab and
                            'standardized_value' in lab and
                            not pd.isna(lab['relative_time'])):
                        all_lab_events.append({
                            'code_index': lab['code_index'],
                            'relative_time': lab['relative_time'],
                            'value': lab['standardized_value']
                        })

        # 按时间排序所有lab事件
        all_lab_events.sort(key=lambda x: x['relative_time'])

        # 提取lab代码、时间和值
        if all_lab_events:
            #lab_codes = [e['code_index'] for e in all_lab_events]
            lab_times = [e['relative_time'] for e in all_lab_events]
            lab_values = [e['value'] for e in all_lab_events]
            # lab类型是100+code_index，确保与其他类型不重合
            lab_types = [100 + e['code_index'] for e in all_lab_events]
        else:
            #lab_codes = []
            lab_times = []
            lab_values = []
            lab_types = []
        # 返回处理后的数据，包括类型信息和新增的lab模态
        return {
            'demographic': demographic_features,
            'medical': {
                'codes': medical_codes,
                'times': medical_times,
                'types': medical_types
            },
            'notes': {
                'embeddings': note_embeddings,
                'times': note_times,
                'types': note_types
            },
            'labs': {
                #'codes': lab_codes,
                'times': lab_times,
                'values': lab_values,
                'types': lab_types
            },
            'label': label,
            'patient_id': patient_data['patient_id']
        }


def custom_collate(batch, fixed_lengths=None):
    """
    自定义collate函数，对不同长度的序列进行填充
    修改版本：处理医疗代码、文本嵌入和lab测试三种模态

    参数:
    batch: 一批次的病人数据
    fixed_lengths: 指定各类事件的固定长度，例如 {
        'medical_events': 600,  # 所有医疗代码合并的总长度
        'note_events': 40,      # 所有文本嵌入合并的总长度
        'lab_events': 200       # 所有lab事件合并的总长度
    }
    """
    if fixed_lengths is None:
        fixed_lengths = {
            'medical_events': 512,  # 所有医疗代码合并在一起
            'note_events': 20,  # 所有文本嵌入合并在一起
            'lab_events': 512  # 所有lab事件合并在一起
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

    # 处理合并的医疗代码
    if 'medical' in batch[0]:
        max_len = fixed_lengths['medical_events']
        codes_batch = []
        times_batch = []
        types_batch = []
        masks_batch = []
        lengths_batch = []

        for item in batch:
            codes = item['medical']['codes']
            times = item['medical']['times']
            types = item['medical']['types']

            # 记录实际长度
            lengths_batch.append(len(codes))

            # 截断或填充
            if len(codes) > max_len:
                # 保留最新的事件
                codes = codes[-max_len:]
                times = times[-max_len:]
                types = types[-max_len:]
                mask = [1] * max_len
            else:
                # 需要填充
                padding_size = max_len - len(codes)
                mask = [1] * len(codes) + [0] * padding_size
                codes = codes + [0] * padding_size
                times = times + [0] * padding_size
                # 为类型填充特殊值'padding'
                types = types + ['padding'] * padding_size

            # 将类型转换为类型ID
            type_ids = []
            for t in types:
                if t == 'diagnosis':
                    type_ids.append(0)
                elif t == 'medication':
                    type_ids.append(1)
                elif t == 'drg':
                    type_ids.append(2)
                else:  # padding
                    type_ids.append(99)

            # 转换为张量
            codes_batch.append(torch.tensor(codes, dtype=torch.long))
            times_batch.append(torch.tensor(times, dtype=torch.float))
            types_batch.append(torch.tensor(type_ids, dtype=torch.long))
            masks_batch.append(torch.tensor(mask, dtype=torch.bool))

        # 合并批次
        result['medical'] = {
            'codes': torch.stack(codes_batch),
            'times': torch.stack(times_batch),
            'types': torch.stack(types_batch),
            'mask': torch.stack(masks_batch),
            'lengths': torch.tensor(lengths_batch, dtype=torch.long)
        }

    # 处理合并的文本嵌入
    if 'notes' in batch[0]:
        max_len = fixed_lengths['note_events']
        emb_batch = []
        times_batch = []
        types_batch = []
        masks_batch = []
        lengths_batch = []

        # 获取嵌入维度
        emb_dim = None
        for item in batch:
            if item['notes']['embeddings'] and len(item['notes']['embeddings']) > 0:
                emb_dim = len(item['notes']['embeddings'][0])
                break

        # 如果没有找到嵌入维度，跳过处理
        if emb_dim is not None:
            for item in batch:
                embeddings = item['notes']['embeddings']
                times = item['notes']['times']
                types = item['notes']['types']

                # 记录实际长度
                lengths_batch.append(len(embeddings))

                # 截断或填充
                if len(embeddings) > max_len:
                    # 保留最新的嵌入
                    embeddings = embeddings[-max_len:]
                    times = times[-max_len:]
                    types = types[-max_len:]
                    mask = [1] * max_len
                else:
                    # 需要填充
                    padding_size = max_len - len(embeddings)
                    mask = [1] * len(embeddings) + [0] * padding_size
                    # 创建零填充的嵌入
                    zero_padding = [np.zeros(emb_dim) for _ in range(padding_size)]
                    embeddings = embeddings + zero_padding
                    times = times + [0] * padding_size
                    # 为类型填充特殊值'padding'
                    types = types + ['padding'] * padding_size

                # 将类型转换为类型ID
                type_ids = []
                for t in types:
                    if t == 'discharge':
                        type_ids.append(3)
                    elif t == 'radiology':
                        type_ids.append(4)
                    else:  # padding
                        type_ids.append(99)
                # 转换为张量
                emb_tensor = torch.tensor(np.array(embeddings), dtype=torch.float)
                times_tensor = torch.tensor(times, dtype=torch.float)
                types_tensor = torch.tensor(type_ids, dtype=torch.long)
                mask_tensor = torch.tensor(mask, dtype=torch.bool)

                emb_batch.append(emb_tensor)
                times_batch.append(times_tensor)
                types_batch.append(types_tensor)
                masks_batch.append(mask_tensor)

            # 合并批次
            result['notes'] = {
                'embeddings': torch.stack(emb_batch) if emb_batch else None,
                'times': torch.stack(times_batch) if times_batch else None,
                'types': torch.stack(types_batch) if types_batch else None,
                'mask': torch.stack(masks_batch) if masks_batch else None,
                'lengths': torch.tensor(lengths_batch, dtype=torch.long) if lengths_batch else None
            }

    # 处理lab事件 - 新增部分
    if 'labs' in batch[0]:
        max_len = fixed_lengths['lab_events']
        codes_batch = []
        times_batch = []
        values_batch = []
        types_batch = []
        masks_batch = []
        lengths_batch = []

        for item in batch:
            #codes = item['labs']['codes']
            times = item['labs']['times']
            values = item['labs']['values']
            types = item['labs']['types']

            # 记录实际长度
            lengths_batch.append(len(times))

            # 截断或填充
            if len(times) > max_len:
                # 保留最新的事件
                #codes = codes[-max_len:]
                times = times[-max_len:]
                values = values[-max_len:]
                types = types[-max_len:]
                mask = [1] * max_len
            else:
                # 需要填充
                padding_size = max_len - len(times)
                mask = [1] * len(times) + [0] * padding_size
                #codes = codes + [0] * padding_size
                times = times + [0] * padding_size
                values = values + [0] * padding_size
                types = types + [99] * padding_size  # 使用99作为填充类型ID

            # 转换为张量
            #codes_batch.append(torch.tensor(codes, dtype=torch.long))
            times_batch.append(torch.tensor(times, dtype=torch.float))
            values_batch.append(torch.tensor(values, dtype=torch.float))
            types_batch.append(torch.tensor(types, dtype=torch.long))
            masks_batch.append(torch.tensor(mask, dtype=torch.bool))
        # 合并批次
        result['labs'] = {
            #'codes': torch.stack(codes_batch),
            'times': torch.stack(times_batch),
            'values': torch.stack(values_batch),
            'types': torch.stack(types_batch),
            'mask': torch.stack(masks_batch),
            'lengths': torch.tensor(lengths_batch, dtype=torch.long)
        }

    return result