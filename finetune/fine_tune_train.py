import pickle
import warnings
import datetime
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, confusion_matrix

warnings.filterwarnings("ignore")
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn as nn

# 导入自定义模块
from mappings import create_all_mappings
from finetune_get_patient_data import PatientDatasetForFinetune, custom_collate_finetune, patient_filter_for_finetune
from model import EHR_Model
from fine_tune_model import *




class Args:
    def __init__(self, batch_size, dx_code_mapping, rx_code_mapping):
        self.batch_size = batch_size
        self.dx_code_mapping = dx_code_mapping
        self.rx_code_mapping = rx_code_mapping
        self.code_vocab_size = 100001
        self.transformer_num_layers = 4  # Transformer层数
        self.transformer_num_head = 8  # 注意力头数
        self.transformer_dim = 256  # 模型维度
        self.dropout = 0.1  # Dropout比率
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_experiment_folder():
    """创建基于时间戳的实验文件夹"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("../experiments_finetune")
    base_dir.mkdir(exist_ok=True)

    # 创建以时间戳命名的实验文件夹
    exp_dir = base_dir / f"experiment_{timestamp}"
    exp_dir.mkdir(exist_ok=True)

    return exp_dir


def evaluate(model, data_loader, device, threshold=0.5):
    """评估模型性能"""
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # 将数据移到设备上
            for key in batch:
                if isinstance(batch[key], dict):
                    for subkey in batch[key]:
                        if isinstance(batch[key][subkey], torch.Tensor):
                            batch[key][subkey] = batch[key][subkey].to(device)
                elif isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # 前向传播
            probs, _ = model(batch)

            # 收集预测和标签
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())

    # 计算评估指标
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # ROC-AUC
    roc_auc = roc_auc_score(all_labels, all_probs)

    # PR-AUC
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)

    # 根据阈值转换为二分类结果
    predictions = (all_probs >= threshold).astype(int)

    # F1分数
    f1 = f1_score(all_labels, predictions)

    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()

    # 计算特异度 (Specificity) 和敏感度 (Sensitivity)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1': f1,
        'specificity': specificity,
        'sensitivity': sensitivity
    }

    return metrics, all_probs, all_labels


def train_readmission_model(pretrained_model_path, train_data_paths, val_data_paths, code_dict, mappings,
                            batch_size, num_epochs, learning_rate, experiment_dir, class_weights=None):
    """
    训练再入院预测模型

    参数:
        pretrained_model_path: 预训练模型路径
        train_data_paths: 训练数据路径
        val_data_paths: 验证数据路径
        code_dict: 代码字典
        mappings: 特征映射
        batch_size: 批次大小
        num_epochs: 训练轮数
        learning_rate: 学习率
        experiment_dir: 实验目录
        class_weights: 类别权重，用于处理类别不平衡
    """
    # 创建模型参数对象
    args = Args(batch_size=batch_size, code_vocab_size=100001)

    # 加载预训练模型
    checkpoint = torch.load(pretrained_model_path, map_location=args.device)

    # 初始化预训练模型
    pretrained_model = EHR_Model(args)
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])

    # 创建再入院预测模型
    model = ReadmissionPredictor(pretrained_model)
    model = model.to(args.device)

    # 定义损失函数（带权重的二元交叉熵）
    if class_weights is not None:
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], device=args.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCELoss()

    # 定义优化器（只优化未冻结的参数）
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # 学习率调整策略
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)

    # 用于早停的变量
    best_val_auc = 0
    early_stop_counter = 0
    patience = 5

    # 获取医疗代码索引集合
    index_set = set(code_dict["index"])

    # 记录每轮训练指标
    train_metrics = []
    val_metrics = []

    for epoch in range(num_epochs):
        epoch_start_time = datetime.datetime.now()
        print(f"\nEpoch {epoch + 1}/{num_epochs} - Starting at {epoch_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 训练阶段
        model.train()
        train_loss = 0
        train_steps = 0

        for file_idx, train_data_path in enumerate(train_data_paths, 1):
            print(f"Processing training file {file_idx}/{len(train_data_paths)}: {train_data_path}")

            with open(train_data_path, 'rb') as f:
                train_data = pickle.load(f)
                # 过滤数据，只保留适合微调的样本
                train_data = patient_filter_for_finetune(train_data)

                # 创建数据集和数据加载器
                dataset = PatientDatasetForFinetune(train_data, mappings, index_set)
                train_loader = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                    collate_fn=custom_collate_finetune
                )

                # 训练循环
                progress_bar = tqdm(train_loader, desc=f"Training")
                for batch_idx, batch in enumerate(progress_bar):
                    # 将所有张量移到设备上
                    for key in batch:
                        if isinstance(batch[key], dict):
                            for subkey in batch[key]:
                                if isinstance(batch[key][subkey], torch.Tensor):
                                    batch[key][subkey] = batch[key][subkey].to(args.device)
                        elif isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(args.device)

                    # 前向传播
                    probs, _ = model(batch)

                    # 计算损失
                    loss = criterion(probs, batch['label'].float())

                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # 更新统计信息
                    train_loss += loss.item()
                    train_steps += 1

                    # 更新进度条
                    progress_bar.set_postfix({'loss': f"{train_loss / train_steps:.4f}"})

        # 计算平均训练损失
        train_loss /= train_steps
        print(f"Training - Loss: {train_loss:.4f}")

        # 验证阶段
        all_val_data = []
        for file_idx, val_data_path in enumerate(val_data_paths, 1):
            print(f"Processing validation file {file_idx}/{len(val_data_paths)}: {val_data_path}")

            with open(val_data_path, 'rb') as f:
                val_data = pickle.load(f)
                val_data = patient_filter_for_finetune(val_data)
                all_val_data.extend(val_data)

        # 创建验证数据集和数据加载器
        val_dataset = PatientDatasetForFinetune(all_val_data, mappings, index_set)
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=custom_collate_finetune
        )

        # 评估模型
        val_metrics_dict, val_probs, val_labels = evaluate(model, val_loader, args.device)

        print(f"Validation - ROC-AUC: {val_metrics_dict['roc_auc']:.4f}, PR-AUC: {val_metrics_dict['pr_auc']:.4f}")
        print(
            f"F1: {val_metrics_dict['f1']:.4f}, Specificity: {val_metrics_dict['specificity']:.4f}, Sensitivity: {val_metrics_dict['sensitivity']:.4f}")

        # 保存指标
        train_metrics.append({'epoch': epoch + 1, 'loss': train_loss})
        val_metrics.append({'epoch': epoch + 1, **val_metrics_dict})

        # 更新学习率
        scheduler.step(val_metrics_dict['roc_auc'])

        # 保存最佳模型
        if val_metrics_dict['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics_dict['roc_auc']
            early_stop_counter = 0

            # 创建保存模型的时间戳
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"model_epoch{epoch + 1}_auc{best_val_auc:.4f}_{timestamp}.pth"
            model_save_path = experiment_dir / model_filename

            # 保存模型
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'timestamp': timestamp
            }, model_save_path)

            print(f"New best model saved to {model_save_path}")

            # 同时保存一个固定名称的"best_model.pth"方便加载
            best_model_path = experiment_dir / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'timestamp': timestamp
            }, best_model_path)
        else:
            early_stop_counter += 1
            print(f"No improvement in validation AUC. Early stop counter: {early_stop_counter}/{patience}")

            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # 保存训练和验证指标为CSV
    pd.DataFrame(train_metrics).to_csv(experiment_dir / "train_metrics.csv", index=False)
    pd.DataFrame(val_metrics).to_csv(experiment_dir / "val_metrics.csv", index=False)

    print(f"\nTraining completed. Results saved to {experiment_dir}")
    return model, best_val_auc


def main():
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 创建实验文件夹
    experiment_dir = create_experiment_folder()
    print(f"Created experiment directory: {experiment_dir}")

    # 定义训练数据文件路径
    train_data_paths = [
        rf"C:\Users\yujun\Desktop\DATA_汇总\data for train v2_病人为单位\data_with_lab\patient_records_batch_{i}.pkl"
        for i in range(1, 15)
    ]

    # 定义验证数据路径
    val_data_paths = [
        rf"C:\Users\yujun\Desktop\DATA_汇总\data for train v2_病人为单位\data_with_lab\patient_records_batch_{i}.pkl"
        for i in range(15, 17)
    ]

    # 定义测试数据路径
    test_data_paths = [
        rf"C:\Users\yujun\Desktop\DATA_汇总\data for train v2_病人为单位\data_with_lab\patient_records_batch_{i}.pkl"
        for i in range(17, 21)
    ]

    # 预训练模型路径
    pretrained_model_path = r"C:\Users\yujun\PycharmProjects\ehr_v2\EHR_v2_初步尝试\ MedM-PLM复现\experiments\experiment_20250313_164108\model_epoch1_valloss0.0044_20250313_164827.pth"

    # 加载人口统计数据
    directory_path = r"C:\Users\yujun\Desktop\NUS学习材料\Nus 科研 EHR\ehr科研告一段落\patien_records_(note_hosp_icu_ed)/patients_dict.csv"
    patient = pd.read_csv(directory_path)
    mappings = create_all_mappings(patient)

    # 加载词嵌入数据
    directory_path = r"C:\Users\yujun\Desktop\DATA_汇总\dict_note_code.parquet"
    code_dict = pd.read_parquet(directory_path)
    index_set = set(code_dict["index"])
    # 超参数
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.0001  # 较小的学习率用于微调

    # 定义类别权重（如果再入院和非再入院样本比例是1:5）
    class_weights = {0: 1.0, 1: 3.0}  # 这些值可能需要根据实际数据集调整

    # 记录训练开始时间
    start_time = datetime.datetime.now()
    print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 训练再入院预测模型
        model, best_val_auc = train_readmission_model(
            pretrained_model_path,
            train_data_paths,
            val_data_paths,
            code_dict,
            mappings,
            batch_size,
            num_epochs,
            learning_rate,
            experiment_dir,
            class_weights
        )

        # 记录训练结束时间
        end_time = datetime.datetime.now()
        print(f"Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Best validation AUC: {best_val_auc:.4f}")

        # 加载最佳模型进行测试
        best_model_path = experiment_dir / "best_model.pth"
        checkpoint = torch.load(best_model_path)

        # 创建预训练模型实例
        args = Args(batch_size=batch_size, code_vocab_size=100001)
        pretrained_model = EHR_Model(args)

        # 创建再入院预测模型实例
        model = ReadmissionPredictor(pretrained_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(args.device)

        # 准备测试数据
        all_test_data = []
        for test_data_path in test_data_paths:
            with open(test_data_path, 'rb') as f:
                test_data = pickle.load(f)
                test_data = patient_filter_for_finetune(test_data)
                all_test_data.extend(test_data)

        # 创建测试数据集和数据加载器
        test_dataset = PatientDatasetForFinetune(all_test_data, mappings, index_set)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=custom_collate_finetune
        )

        # 在测试集上评估模型
        test_metrics, test_probs, test_labels = evaluate(model, test_loader, args.device)

        print(f"\nTest Results:")
        print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}, PR-AUC: {test_metrics['pr_auc']:.4f}")
        print(
            f"F1: {test_metrics['f1']:.4f}, Specificity: {test_metrics['specificity']:.4f}, Sensitivity: {test_metrics['sensitivity']:.4f}")

        # 保存测试结果
        pd.DataFrame({
            'label': test_labels,
            'probability': test_probs
        }).to_csv(experiment_dir / "test_predictions.csv", index=False)

        # 保存测试指标
        pd.DataFrame([test_metrics]).to_csv(experiment_dir / "test_metrics.csv", index=False)

    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e


if __name__ == "__main__":
    main()