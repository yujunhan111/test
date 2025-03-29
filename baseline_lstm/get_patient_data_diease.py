from tool import get_history_codes
from torch.utils.data import Dataset, DataLoader
from filter_patients import filter_valid_patients
import numpy as np
import torch
import pandas as pd
from disease_codes import DISEASE_CODES


class PatientDataset_disease(Dataset):
    POSITIVE_LABEL = 0.95
    NEGATIVE_LABEL = 0.05
    UNDEFINED_LABEL = -1
    disease_codes = DISEASE_CODES

    def __init__(self, data, mappings, index_set):
        self.patient_ids = list(data.keys())
        self.data = data
        self.mappings = mappings
        self.index_set = index_set
        self.max_indices = {
            'gender': max(v for v in mappings['gender'].values() if isinstance(v, (int, float))),
            'race': max(v for v in mappings['race'].values() if isinstance(v, (int, float))),
            'marital_status': max(v for v in mappings['marital_status'].values() if isinstance(v, (int, float))),
            'language': max(v for v in mappings['language'].values() if isinstance(v, (int, float)))
        }
        print(f"Dataset initialized with {len(self.patient_ids)} patients")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        try:
            patient_id = self.patient_ids[idx]
            patient_data = self.data[patient_id]
            demo = patient_data['demographics']
            features = [1.0]
            features.append(np.log1p(float(demo['age'])))
            for feature in ['gender', 'race', 'marital_status', 'language']:
                feature_value = demo[feature]
                if pd.isna(feature_value):
                    feature_value = 'nan'
                feat_idx = self.mappings[feature].get(feature_value, self.mappings[feature]['nan'])
                one_hot = [0] * (self.max_indices[feature] + 1)
                one_hot[feat_idx] = 1
                features.extend(one_hot)
            demographic_features = torch.tensor(features, dtype=torch.float32)

            visits = []
            for event in patient_data['events']:
                if event['codes']:
                    visit_codes = [(code['code_index'], np.log1p(code['time'] / (24 * 7)))
                                   for code in event['codes']
                                   if code['code_index'] in self.index_set and not pd.isna(code['time'])]
                    if visit_codes:
                        visits.append(visit_codes)
            visits = sorted(visits, key=lambda x: x[0][1])  # 确保按时间排序

            disease_data = {}
            for disease_name, disease_codes_list in self.disease_codes.items():
                disease_codes_set = set(disease_codes_list)
                occurrence_count = 0
                first_occurrence_visit_start = None

                for visit in visits:
                    visit_codes_set = {code_idx for code_idx, _ in visit}
                    matching_codes = visit_codes_set & disease_codes_set
                    if matching_codes:
                        occurrence_count += 1
                        if first_occurrence_visit_start is None:
                            first_occurrence_visit_start = visit[0][1]

                if occurrence_count >= 2:
                    history = get_history_codes(visits, first_occurrence_visit_start, self.index_set)
                    if history:
                        disease_data[disease_name] = {
                            'label': 0.95,
                            'event_time': first_occurrence_visit_start,
                            'history_codes': history[0],
                            'history_times': history[1]
                        }
                elif occurrence_count == 0:
                    eval_time = visits[-1][0][1]
                    history = get_history_codes(visits, eval_time, self.index_set)
                    if history:
                        disease_data[disease_name] = {
                            'label': 0.05,
                            'event_time': eval_time,
                            'history_codes': history[0],
                            'history_times': history[1]
                        }
                else:
                    if first_occurrence_visit_start is not None:
                        history = get_history_codes(visits, first_occurrence_visit_start, self.index_set)
                        if history:
                            disease_data[disease_name] = {
                                'label': self.UNDEFINED_LABEL,
                                'event_time': first_occurrence_visit_start,
                                'history_codes': history[0],
                                'history_times': history[1]
                            }

            return {
                'demographic': demographic_features,
                'disease_data': disease_data
            }
        except Exception as e:
            print(f"Error loading patient {idx}: {str(e)}")
            raise e