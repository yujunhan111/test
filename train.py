import pickle
import warnings
import numpy as np
from pytorch_toolbelt.losses import BinaryFocalLoss
import os
import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import torch

# 导入自定义模块
from mappings import create_all_mappings
from get_patient_data import PatientDataset, custom_collate
from transformer import *
from model import EHR_MBT_Model, handle_missing_modality
from utils import *

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


def create_experiment_folder():
    """创建基于时间戳的实验文件夹"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("./experiments")
    base_dir.mkdir(exist_ok=True)

    # 创建以时间戳命名的实验文件夹
    exp_dir = base_dir / f"experiment_{timestamp}"
    exp_dir.mkdir(exist_ok=True)

    return exp_dir


def train_model(train_data_paths, val_data_paths, code_dict, mappings, batch_size, num_epochs, learning_rate,
                gradient_accumulation_steps, patience, experiment_dir, use_modal_augmentation=True, max_augmentation_prob=0.3, curriculum_epochs=10):
    # 创建模型参数对象
    args = Args(batch_size=batch_size, code_vocab_size=12233)

    # 创建模型
    model = EHR_MBT_Model(args, code_dict=code_dict)
    model = model.to(args.device)

    # 定义损失函数和优化器
    criterion = BinaryFocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience // 2, factor=0.5)

    # 用于早停的变量
    best_val_auc = 0
    early_stop_counter = 0

    # 获取医疗代码索引集合
    index_set = set(code_dict["index"])

    for epoch in range(num_epochs):
        epoch_start_time = datetime.datetime.now()
        print(f"\nEpoch {epoch + 1}/{num_epochs} - Starting at {epoch_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 训练阶段
        model.train()
        current_aug_prob = 0.0
        if use_modal_augmentation:
            if epoch >= curriculum_epochs:
                current_aug_prob = max_augmentation_prob
            else:
                # 逐步增加增强概率
                current_aug_prob = (epoch / curriculum_epochs) * max_augmentation_prob
        print(f"Current modal masking probability: {current_aug_prob:.4f}")

        train_loss = 0
        train_steps = 0
        all_preds = []
        all_labels = []

        for file_idx, train_data_path in enumerate(train_data_paths, 1):
            print(f"Processing training file {file_idx}/{len(train_data_paths)}: {train_data_path}")

            with open(train_data_path, 'rb') as f:
                train_data = pickle.load(f)
                dataset = PatientDataset(train_data, mappings, index_set)
                train_loader = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    collate_fn=custom_collate
                )

                # 训练循环
                progress_bar = tqdm(train_loader, desc=f"Training")
                for batch_idx, batch in enumerate(progress_bar):
                    if use_modal_augmentation and current_aug_prob > 0:
                        batch = apply_modal_augmentation(batch, current_aug_prob, args.device)
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

                    # 前向传播（包含missing参数）
                    outputs = model(batch, missing=missing)

                    # 计算损失
                    labels = batch['label'].float().unsqueeze(1)
                    loss = criterion(outputs, labels)

                    # 缩放损失以适应梯度累积
                    loss = loss / gradient_accumulation_steps

                    # 反向传播
                    loss.backward()

                    # 梯度累积
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    # 更新统计信息
                    train_loss += loss.item() * gradient_accumulation_steps
                    train_steps += 1

                    # 收集预测和标签用于计算指标
                    probs = torch.sigmoid(outputs).detach().cpu().numpy()
                    all_preds.extend(probs)
                    all_labels.extend(labels.detach().cpu().numpy())

                    # 更新进度条
                    progress_bar.set_postfix({'loss': f"{train_loss / train_steps:.4f}"})

        # 计算训练指标
        train_loss /= train_steps
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        train_auc = roc_auc_score(all_labels, all_preds)
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        train_auprc = auc(recall, precision)

        print(f"Training - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, AUPRC: {train_auprc:.4f}")

        # 验证阶段
        model.eval()
        val_loss = 0
        val_steps = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for file_idx, val_data_path in enumerate(val_data_paths, 1):
                print(f"Processing validation file {file_idx}/{len(val_data_paths)}: {val_data_path}")

                with open(val_data_path, 'rb') as f:
                    val_data = pickle.load(f)
                    dataset = PatientDataset(val_data, mappings, index_set)
                    val_loader = DataLoader(
                        dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True,
                        collate_fn=custom_collate
                    )

                    # 验证循环
                    progress_bar = tqdm(val_loader, desc=f"Validation")
                    for batch_idx, batch in enumerate(progress_bar):
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

                        # 前向传播（包含missing参数）
                        outputs = model(batch, missing=missing)

                        # 计算损失
                        labels = batch['label'].float().unsqueeze(1)
                        loss = criterion(outputs, labels)

                        # 更新统计信息
                        val_loss += loss.item()
                        val_steps += 1

                        # 收集预测和标签
                        probs = torch.sigmoid(outputs).cpu().numpy()
                        all_val_preds.extend(probs)
                        all_val_labels.extend(labels.cpu().numpy())

                        # 更新进度条
                        progress_bar.set_postfix({'loss': f"{val_loss / val_steps:.4f}"})

        # 计算验证指标
        val_loss /= val_steps
        all_val_preds = np.array(all_val_preds).flatten()
        all_val_labels = np.array(all_val_labels).flatten()
        val_auc = roc_auc_score(all_val_labels, all_val_preds)
        precision, recall, _ = precision_recall_curve(all_val_labels, all_val_preds)
        val_auprc = auc(recall, precision)

        print(f"Validation - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, AUPRC: {val_auprc:.4f}")

        # 更新学习率
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            early_stop_counter = 0

            # 创建保存模型的时间戳
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"model_epoch{epoch + 1}_valauc{val_auc:.4f}_{timestamp}.pth"
            model_save_path = experiment_dir / model_filename

            # 保存模型
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'timestamp': timestamp
            }, model_save_path)

            print(f"New best model saved to {model_save_path}")

            # 同时保存一个固定名称的"best_model.pth"方便加载
            best_model_path = experiment_dir / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'timestamp': timestamp
            }, best_model_path)
        else:
            early_stop_counter += 1
            print(f"No improvement in validation AUC. Early stop counter: {early_stop_counter}/{patience}")

            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    print(f"\nTraining completed. Results saved to {experiment_dir}")
    return model, best_val_auc


def main(batch_size, num_epochs, learning_rate, gradient_accumulation_steps, patience):
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 创建实验文件夹
    experiment_dir = create_experiment_folder()
    print(f"Created experiment directory: {experiment_dir}")

    # 定义训练数据文件路径
    train_data_paths = [
        rf"C:\Users\yujun\Desktop\DATA_汇总\data for train v2_病人为单位\data_with_lab\patient_records_batch_{i}.pkl" for i in
        range(1, 15)]

    # 定义验证数据路径
    val_data_paths = [
        rf"C:\Users\yujun\Desktop\DATA_汇总\data for train v2_病人为单位\data_with_lab\patient_records_batch_{i}.pkl" for i in
        range(15, 17)]
    # 加载人口统计数据
    directory_path = r"C:\Users\yujun\Desktop\NUS学习材料\Nus 科研 EHR\ehr科研告一段落\patien_records_(note_hosp_icu_ed)/patients_dict.csv"
    patient = pd.read_csv(directory_path)
    mappings = create_all_mappings(patient)

    # 加载词嵌入数据
    directory_path = r"C:\Users\yujun\Desktop\DATA_汇总\dict_note_code.parquet"
    code_dict = pd.read_parquet(directory_path)

    # 记录训练开始时间
    start_time = datetime.datetime.now()
    print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        model, best_val_auc = train_model(
            train_data_paths,
            val_data_paths,
            code_dict,
            mappings,
            batch_size,
            num_epochs,
            learning_rate,
            gradient_accumulation_steps,
            patience,
            experiment_dir
        )

        # 记录训练结束时间
        end_time = datetime.datetime.now()
        print(f"Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Best validation AUC: {best_val_auc:.4f}")

    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e


if __name__ == "__main__":
    main(batch_size=32, num_epochs=200, learning_rate=0.0005, gradient_accumulation_steps=1, patience=5)