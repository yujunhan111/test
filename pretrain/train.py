import pickle
import warnings
import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch

# 导入自定义模块
from mappings import create_all_mappings
from get_patient_data import PatientDataset, custom_collate
from model import EHR_Model
from utils import *

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
    base_dir = Path("../../MedM-PLM复现_visit_单位/experiments")
    base_dir.mkdir(exist_ok=True)

    # 创建以时间戳命名的实验文件夹
    exp_dir = base_dir / f"experiment_{timestamp}"
    exp_dir.mkdir(exist_ok=True)

    return exp_dir

def train_multimodal_pretraining(
    train_data_paths, val_data_paths, code_dict, mappings, code_mappings,
    batch_size, num_epochs, learning_rate, gradient_accumulation_steps, patience, experiment_dir
):
    # 创建模型参数对象
    args = Args(
        batch_size=batch_size,
        dx_code_mapping=code_mappings['dx_code_mapping'],
        rx_code_mapping=code_mappings['rx_code_mapping']
    )

    # 创建模型
    model = EHR_Model(args)
    model = model.to(args.device)

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience // 2, factor=0.5)

    # 用于早停的变量
    best_val_loss = float('inf')
    early_stop_counter = 0

    # 获取医疗代码索引集合
    index_set = set(code_dict["index"])

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
                visit_data = convert_to_visit_level(train_data)
                visit_data = patient_filter(visit_data)
                dataset = PatientDataset(visit_data, mappings,code_mappings, index_set, True, 0.15)
                sample = dataset[0]
                print("Diagnosis Codes (after dedup):", sample['diagnosis']['codes'])
                print("Original Diagnosis Codes:", sample['original_codes']['diagnosis'])
                train_loader = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                    collate_fn=custom_collate
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

                    # 前向传播（预训练模式）
                    loss = model(batch)

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

                    # 更新进度条
                    progress_bar.set_postfix({'loss': f"{train_loss / train_steps:.4f}"})

        # 计算平均训练损失
        train_loss /= train_steps
        print(f"Training - Loss: {train_loss:.4f}")

        # 验证阶段
        model.eval()
        val_loss = 0
        val_steps = 0

        with torch.no_grad():
            for file_idx, val_data_path in enumerate(val_data_paths, 1):
                print(f"Processing validation file {file_idx}/{len(val_data_paths)}: {val_data_path}")

                with open(val_data_path, 'rb') as f:
                    val_data = pickle.load(f)
                    # 转换为visit级别
                    val_data = convert_to_visit_level(val_data)
                    # 过滤visit数据
                    val_data = patient_filter(val_data)
                    # 创建数据集
                    dataset = PatientDataset(val_data, mappings,code_mappings, index_set, True, 0.15)
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

                        # 前向传播
                        loss = model(batch)

                        # 更新统计信息
                        val_loss += loss.item()
                        val_steps += 1

                        # 更新进度条
                        progress_bar.set_postfix({'loss': f"{val_loss / val_steps:.4f}"})

        # 计算平均验证损失
        val_loss /= val_steps
        print(f"Validation - Loss: {val_loss:.4f}")

        # 更新学习率
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0

            # 创建保存模型的时间戳
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"model_epoch{epoch + 1}_valloss{val_loss:.4f}_{timestamp}.pth"
            model_save_path = experiment_dir / model_filename

            # 保存模型
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'timestamp': timestamp
            }, model_save_path)

            print(f"New best model saved to {model_save_path}")

            # 同时保存一个固定名称的"best_model.pth"方便加载
            best_model_path = experiment_dir / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'timestamp': timestamp
            }, best_model_path)
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss. Early stop counter: {early_stop_counter}/{patience}")

            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    print(f"\nTraining completed. Results saved to {experiment_dir}")
    return model, best_val_loss

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
        rf"C:\Users\yujun\Desktop\DATA_汇总\data for train v2_病人为单位\data_with_lab\patient_records_batch_{i}.pkl"
        for i in range(1, 15)
    ]

    # 定义验证数据路径
    val_data_paths = [
        rf"C:\Users\yujun\Desktop\DATA_汇总\data for train v2_病人为单位\data_with_lab\patient_records_batch_{i}.pkl"
        for i in range(15, 17)
    ]

    # 加载人口统计数据
    directory_path = r"C:\Users\yujun\Desktop\NUS学习材料\Nus 科研 EHR\ehr科研告一段落\patien_records_(note_hosp_icu_ed)/patients_dict.csv"
    patient = pd.read_csv(directory_path)
    mappings = create_all_mappings(patient)

    # 加载词嵌入数据
    directory_path = r"C:\Users\yujun\Desktop\DATA_汇总\dict_note_code.parquet"
    code_dict = pd.read_parquet(directory_path)

    # 加载代码映射
    with open(r"C:\Users\yujun\Downloads\rx_mapping.pkl", 'rb') as f:
        rx_code_mapping = pickle.load(f)
    with open(r"C:\Users\yujun\Downloads\dx_mapping.pkl", 'rb') as f:
        dx_code_mapping = pickle.load(f)
    code_mappings = {'dx_code_mapping': dx_code_mapping, 'rx_code_mapping': rx_code_mapping}

    # 记录训练开始时间
    start_time = datetime.datetime.now()
    print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 使用多模态预训练函数
        model, best_val_loss = train_multimodal_pretraining(
            train_data_paths,
            val_data_paths,
            code_dict,
            mappings,
            code_mappings,
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
        print(f"Best validation loss: {best_val_loss:.4f}")

    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

if __name__ == "__main__":
    main(batch_size=64, num_epochs=200, learning_rate=0.0005, gradient_accumulation_steps=1, patience=5)