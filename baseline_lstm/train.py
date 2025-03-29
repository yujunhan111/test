import pickle
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
from tqdm import tqdm
import gc
import json
from datetime import datetime
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from LSTM import LSTMModel
from get_patient_data_diease import PatientDataset_disease
from custom_collate_disease import custom_collate_disease
from mappings import create_clean_mapping, create_all_mappings
from disease_codes import DISEASE_CODES, disease_weights
from filter_patients import filter_valid_patients
from evaluation_disease import evaluate_model_disease, evaluate_epoch, print_metrics_summary


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_auroc = None
        self.early_stop = False
        self.best_model = None
        self.improvement_epochs = []  # 记录性能提升的epochs

    def __call__(self, val_auroc, model, epoch):
        if self.best_auroc is None:
            self.best_auroc = val_auroc
            self.improvement_epochs.append(epoch)
            self.save_checkpoint(model)
            return False

        if val_auroc > self.best_auroc + self.min_delta:
            self.best_auroc = val_auroc
            self.improvement_epochs.append(epoch)
            self.save_checkpoint(model)
            self.counter = 0
            return False
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

    def save_checkpoint(self, model):
        """保存最佳模型状态"""
        import copy
        self.best_model = copy.deepcopy(model.state_dict())


def create_dataloader(dataset, batch_size=1):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_disease
    )


def train_disease_model(train_data_paths, val_data_paths, mappings, batch_size,
                        num_epochs, learning_rate, gradient_accumulation_steps, patience):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载 code_dict
    code_dict_path = r"C:\Users\yujun\PycharmProjects\ehr_迁移\pic\data\code-dict-with-embedding-pic.parquet"
    code_dict = pd.read_parquet(code_dict_path)

    # 提取 index_set
    if 'index' not in code_dict.columns:
        raise ValueError("code_dict 中缺少 'index' 列，请检查文件内容。")
    index_set = set(code_dict['index'].astype(int))  # 确保 index 是整数类型
    print(f"Loaded index_set with {len(index_set)} unique indices.")


    num_codes = 3000

    # 初始化模型，直接用编码器处理code
    model = LSTMModel(
        num_codes=num_codes,
        hidden_dim=128,
        disease_names=list(DISEASE_CODES.keys()),
        demographic_dim=70
    ).to(device)

    combined_weights = {**disease_weights}
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_avg_auroc = 0
    no_improvement = 0
    improvement_epochs = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        total_batches = 0

        # 训练循环
        for file_idx, train_path in enumerate(train_data_paths, 1):
            try:
                print(f"\nProcessing file {file_idx}/{len(train_data_paths)}: {train_path}")

                # 加载和过滤数据
                with open(train_path, 'rb') as f:
                    current_data = pickle.load(f)
                filtered_data = filter_valid_patients(current_data)
                print(f"Filtered patients: {len(filtered_data)} / {len(current_data)}")

                # 创建训练数据集，传递 index_set，直接传入字典
                train_dataset = PatientDataset_disease(filtered_data, mappings, index_set)
                train_loader = create_dataloader(train_dataset, batch_size)

                running_loss = 0.0
                n_batches = 0

                progress_bar = tqdm(total=len(filtered_data),
                                    desc=f"Processing patients in file {file_idx}",
                                    unit="patient",
                                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, Loss: {postfix[0]:.4f}]',
                                    postfix=[0.0])

                for batch_idx, batch in enumerate(train_loader):
                    try:
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                 for k, v in batch.items()}

                        batch_loss = model(batch, disease_weights=combined_weights, is_training=True)

                        if batch_loss > 0:  # 只在有有效样本时更新
                            batch_loss = batch_loss / gradient_accumulation_steps
                            batch_loss.backward()

                            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                optimizer.step()
                                optimizer.zero_grad()

                            running_loss += batch_loss.item()
                            n_batches += 1
                            current_avg_loss = running_loss / n_batches
                            progress_bar.postfix[0] = current_avg_loss
                            epoch_loss += batch_loss.item()
                            total_batches += 1

                        progress_bar.update(batch_size)

                    except Exception as e:
                        print(f"Error in batch processing: {str(e)}")
                        continue

                progress_bar.close()
                del train_dataset, train_loader, filtered_data
                gc.collect()

            except Exception as e:
                print(f"Error processing file {train_path}: {str(e)}")
                continue

        # 验证阶段
        print("\nStarting validation...")
        model.eval()

        # 加载验证数据
        val_data = {}
        for val_path in val_data_paths:
            with open(val_path, 'rb') as f:
                val_data.update(pickle.load(f))
        filtered_val_data = filter_valid_patients(val_data)
        filtered_val_data = dict(list(filtered_val_data.items())[:500])  # 限制数量
        # 创建验证数据集，传递 index_set，直接传入字典
        val_dataset = PatientDataset_disease(filtered_val_data, mappings, index_set)
        val_loader = create_dataloader(val_dataset, batch_size)

        val_metrics = evaluate_epoch(model, val_loader, device)
        # 打印详细的验证结果
        print_metrics_summary(val_metrics)
        # 计算平均AUROC（仅考虑疾病预测）
        disease_aurocs = [
            metrics['auroc']
            for disease, metrics in val_metrics['disease_metrics'].items()
            if disease in DISEASE_CODES and not np.isnan(metrics['auroc'])
        ]
        current_avg_auroc = np.mean(disease_aurocs) if disease_aurocs else 0

        print(f"\nEpoch {epoch + 1}")
        print(f"Current Average AUROC: {current_avg_auroc:.4f}")
        print(f"Best Average AUROC: {best_avg_auroc:.4f}")

        # 检查是否有性能提升
        if current_avg_auroc > best_avg_auroc:
            best_avg_auroc = current_avg_auroc
            improvement_epochs.append(epoch + 1)
            no_improvement = 0

            # 只在性能提升时保存模型
            save_path = f'dipole_model_epoch_{epoch + 1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_auroc': current_avg_auroc,
                'val_metrics': val_metrics
            }, save_path)
            print(f"Model saved at epoch {epoch + 1} with improved AUROC: {current_avg_auroc:.4f}")
        else:
            no_improvement += 1
            print(f"No improvement for {no_improvement} epochs")

        # Early stopping 检查
        if no_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            print(f"Best AUROC achieved: {best_avg_auroc:.4f}")
            print(f"Performance improved at epochs: {improvement_epochs}")
            break

        # 清理验证数据
        del val_dataset, val_loader, filtered_val_data, val_data
        gc.collect()

    return model, best_avg_auroc, improvement_epochs


def main(batch_size, num_epochs, learning_rate, gradient_accumulation_steps, patience):
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_data_paths = [rf"C:\Users\yujun\Desktop\patient_records_pic\patient_records_batch_{i}.pkl" for i in
                        range(1, 2)]
    val_data_paths = [
        r"C:\Users\yujun\Desktop\patient_records_pic\patient_records_batch_8.pkl",
    ]
    directory_path = r"C:\Users\yujun\PycharmProjects\ehr_迁移\pic\数据预处理\records\patients_dict.csv"
    patient = pd.read_csv(directory_path)
    mappings = create_all_mappings(patient)

    train_disease_model(train_data_paths, val_data_paths, mappings, batch_size, num_epochs, learning_rate,
                        gradient_accumulation_steps, patience)


if __name__ == "__main__":
    main(batch_size=1, num_epochs=200, learning_rate=0.0001, gradient_accumulation_steps=16, patience=3)