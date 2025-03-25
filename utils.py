import numpy as np
import torch
import random

def apply_modal_augmentation(batch, aug_prob, device):
    """
    对批次数据应用模态掩蔽数据增强

    参数:
    batch: 批次数据
    aug_prob: 应用增强的概率
    device: 计算设备

    返回:
    augmented_batch: 增强后的批次数据
    """
    batch_size = batch['demographic'].size(0)

    # 为每个样本决定是否应用增强
    for i in range(batch_size):
        # 有aug_prob的概率应用增强
        if random.random() < aug_prob:
            # 随机决定掩蔽哪些模态
            # 按不同的概率掩蔽不同的模态组合
            modality_mask = random.choices(
                [
                    [1, 0, 1],  # 掩蔽note
                    [1, 1, 0],  # 掩蔽lab
                    [0, 1, 1],  # 掩蔽medical
                    [0, 0, 1],  # 掩蔽medical和note
                    [0, 1, 0],  # 掩蔽medical和lab
                    [1, 0, 0],  # 掩蔽note和lab
                ],
                weights=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1],  # 权重可以根据需要调整

                k=1
            )[0]

            # 应用掩蔽 - 医疗代码
            if modality_mask[0] == 0:
                batch['medical']['mask'][i] = torch.zeros_like(batch['medical']['mask'][i])
                batch['medical']['lengths'][i] = 0  # 同时更新lengths

            # 应用掩蔽 - 注释
            if modality_mask[1] == 0 and 'notes' in batch and batch['notes']['mask'] is not None:
                batch['notes']['mask'][i] = torch.zeros_like(batch['notes']['mask'][i])
                batch['notes']['lengths'][i] = 0  # 同时更新lengths

            # 应用掩蔽 - lab
            if modality_mask[2] == 0 and 'labs' in batch and batch['labs']['mask'] is not None:
                batch['labs']['mask'][i] = torch.zeros_like(batch['labs']['mask'][i])
                batch['labs']['lengths'][i] = 0  # 同时更新lengths

    return batch
def check_batch(batch, show_samples=2, show_items=5):
    """
    检查批处理数据，显示填充和真实数据的统计信息

    参数:
    batch: custom_collate处理后的批次数据
    show_samples: 显示的样本数量
    show_items: 每个样本显示的项数
    """
    print("\n==== 批次数据检查 ====")

    batch_size = len(batch['patient_id']) if 'patient_id' in batch else batch['label'].size(0)
    print(f"批次大小: {batch_size}")

    # 显示部分患者ID
    if 'patient_id' in batch:
        print(f"患者ID: {batch['patient_id'][:show_samples]}")

    # 检查demographic特征
    if 'demographic' in batch:
        print(f"Demographic特征形状: {batch['demographic'].shape}")

    # 检查标签
    if 'label' in batch:
        print(f"标签: {batch['label']}")

    # 检查医疗代码和时间
    code_keys = ['diagnosis', 'medication', 'drg', 'lab']
    for key in code_keys:
        if key in batch:
            print(f"\n{key.capitalize()}:")
            codes = batch[key]['codes']
            times = batch[key]['times']
            masks = batch[key]['mask']

            print(f"  形状: {codes.shape}")

            # 计算每个样本真实数据的数量
            real_counts = masks.sum(dim=1).tolist()
            print(f"  真实数据数量: {real_counts[:show_samples]}")

            # 计算填充率
            padding_rates = [(1 - count / codes.shape[1]) * 100 for count in real_counts]
            print(f"  填充率(%): {[f'{rate:.2f}%' for rate in padding_rates[:show_samples]]}")

            # 显示部分样本的代码和时间
            for i in range(min(show_samples, batch_size)):
                real_count = real_counts[i]
                print(f"  样本 {i} (真实数据: {real_count}, 填充率: {padding_rates[i]:.2f}%):")

                # 显示真实数据的一部分
                if real_count > 0:
                    show_count = min(show_items, real_count)
                    print(f"    真实代码前{show_count}项: {codes[i, :show_count].tolist()}")
                    print(f"    真实时间前{show_count}项: {times[i, :show_count].tolist()}")

                # 如果有填充数据，显示部分填充数据
                if codes.shape[1] > real_count:
                    padding_start = real_count
                    padding_end = min(padding_start + show_items, codes.shape[1])
                    pad_show_count = padding_end - padding_start
                    if pad_show_count > 0:
                        print(f"    填充代码前{pad_show_count}项: {codes[i, padding_start:padding_end].tolist()}")
                        print(f"    填充时间前{pad_show_count}项: {times[i, padding_start:padding_end].tolist()}")

    # 检查note嵌入
    emb_keys = ['dis_embeddings', 'rad_embeddings']
    for key in emb_keys:
        if key in batch and batch[key]['embeddings'] is not None:
            print(f"\n{key}:")
            embeddings = batch[key]['embeddings']
            times = batch[key]['times']
            masks = batch[key]['mask']

            print(f"  形状: {embeddings.shape}")

            # 计算每个样本真实嵌入的数量
            real_counts = masks.sum(dim=1).tolist()
            print(f"  真实嵌入数量: {real_counts[:show_samples]}")

            # 计算填充率
            padding_rates = [(1 - count / embeddings.shape[1]) * 100 for count in real_counts]
            print(f"  填充率(%): {[f'{rate:.2f}%' for rate in padding_rates[:show_samples]]}")

            # 显示部分样本的嵌入形状和时间
            for i in range(min(show_samples, batch_size)):
                real_count = real_counts[i]
                print(f"  样本 {i} (真实嵌入: {real_count}, 填充率: {padding_rates[i]:.2f}%):")

                # 显示真实嵌入的形状
                if real_count > 0:
                    show_count = min(show_items, real_count)
                    for j in range(show_count):
                        print(f"    嵌入 {j} 形状: {embeddings[i, j].shape}")
                    print(f"    真实时间前{show_count}项: {times[i, :show_count].tolist()}")