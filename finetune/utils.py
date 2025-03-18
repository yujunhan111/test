import random
import numpy as np
import copy
def check_batch(batch, show_samples=2, show_items=5):
    """
    检查批处理数据，显示填充和真实数据的统计信息
    适用于多模态数据（含结构化医疗代码、文本嵌入和实验室检测数据）

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
        print(f"标签: {batch['label'][:show_samples].tolist()}")

    # 结构化医疗代码模态 (诊断、药物、DRG)
    medical_code_keys = ['diagnosis', 'medication', 'drg']
    for key in medical_code_keys:
        if key in batch:
            print(f"\n=== {key.capitalize()} 代码 ===")

            codes = batch[key]['codes']
            times = batch[key]['times']
            masks = batch[key]['mask']
            lengths = batch[key]['lengths'] if 'lengths' in batch[key] else masks.sum(dim=1)

            print(f"  形状: {codes.shape}")
            print(f"  真实数据数量: {lengths[:show_samples].tolist()}")

            # 计算填充率（确保不会出现负数）
            padding_rates = []
            for length in lengths[:show_samples]:
                actual_length = min(length.item(), codes.shape[1])  # 实际使用的长度不能超过最大长度
                rate = (1 - actual_length / codes.shape[1]) * 100
                padding_rates.append(rate)
            print(f"  填充率(%): {[f'{rate:.2f}%' for rate in padding_rates]}")

            # 显示部分样本的代码和时间
            for i in range(min(show_samples, batch_size)):
                real_count = lengths[i].item()
                # 为显示目的修正实际长度值
                display_count = real_count
                if real_count > codes.shape[1]:
                    print(f"  注意：样本 {i} 实际有 {real_count} 项数据，超出了最大长度 {codes.shape[1]}，已被截断")
                print(f"  样本 {i} (真实数据: {real_count}, 填充率: {padding_rates[i]:.2f}%):")

                # 显示真实数据的一部分
                if real_count > 0:
                    show_count = min(show_items, real_count)
                    print(f"    真实代码前{show_count}项: {codes[i, :show_count].tolist()}")
                    print(f"    真实时间前{show_count}项: {times[i, :show_count].tolist()}")

                # 显示部分填充数据
                if codes.shape[1] > real_count:
                    padding_start = real_count
                    padding_end = min(padding_start + show_items, codes.shape[1])
                    pad_show_count = padding_end - padding_start
                    if pad_show_count > 0:
                        print(f"    填充代码前{pad_show_count}项: {codes[i, padding_start:padding_end].tolist()}")
                        print(f"    填充时间前{pad_show_count}项: {times[i, padding_start:padding_end].tolist()}")

    # 文本嵌入模态 (出院摘要、放射学报告嵌入)
    text_emb_keys = ['discharge_summary', 'radiology_report']
    for key in text_emb_keys:
        if key in batch and 'embeddings' in batch[key] and batch[key]['embeddings'] is not None:
            print(f"\n=== {key.replace('_', ' ').title()} 嵌入 ===")

            embeddings = batch[key]['embeddings']
            times = batch[key]['times']
            masks = batch[key]['mask']
            lengths = batch[key]['lengths'] if 'lengths' in batch[key] else masks.sum(dim=1)

            print(f"  形状: {embeddings.shape}")
            print(f"  嵌入维度: {embeddings.shape[-1]}")
            print(f"  真实嵌入数量: {lengths[:show_samples].tolist()}")

            # 计算填充率（确保不会出现负数）
            padding_rates = []
            for length in lengths[:show_samples]:
                actual_length = min(length.item(), embeddings.shape[1])  # 实际使用的长度不能超过最大长度
                rate = (1 - actual_length / embeddings.shape[1]) * 100
                padding_rates.append(rate)
            print(f"  填充率(%): {[f'{rate:.2f}%' for rate in padding_rates]}")

            # 显示部分样本的嵌入和时间
            for i in range(min(show_samples, batch_size)):
                real_count = lengths[i].item()
                print(f"  样本 {i} (真实嵌入: {real_count}, 填充率: {padding_rates[i]:.2f}%):")

                # 显示真实嵌入的信息
                if real_count > 0:
                    show_count = min(show_items, real_count)
                    for j in range(show_count):
                        print(f"    嵌入 {j} 形状: {embeddings[i, j].shape}")
                        # 显示嵌入向量的一小部分值
                        print(f"    嵌入 {j} 前5个值: {embeddings[i, j, :5].tolist()}")
                    print(f"    真实时间前{show_count}项: {times[i, :show_count].tolist()}")

    # 文本代码模态 (出院摘要、放射学报告代码)
    text_code_keys = ['discharge_codes', 'radiology_codes']
    for key in text_code_keys:
        if key in batch:
            print(f"\n=== {key.replace('_', ' ').title()} ===")

            codes = batch[key]['codes']
            times = batch[key]['times']
            masks = batch[key]['mask']
            lengths = batch[key]['lengths'] if 'lengths' in batch[key] else masks.sum(dim=1)

            print(f"  形状: {codes.shape}")
            print(f"  真实数据数量: {lengths[:show_samples].tolist()}")

            # 计算填充率（确保不会出现负数）
            padding_rates = []
            for length in lengths[:show_samples]:
                actual_length = min(length.item(), codes.shape[1])  # 实际使用的长度不能超过最大长度
                rate = (1 - actual_length / codes.shape[1]) * 100
                padding_rates.append(rate)
            print(f"  填充率(%): {[f'{rate:.2f}%' for rate in padding_rates]}")

            # 显示部分样本的代码和时间
            for i in range(min(show_samples, batch_size)):
                real_count = lengths[i].item()
                print(f"  样本 {i} (真实数据: {real_count}, 填充率: {padding_rates[i]:.2f}%):")

                # 显示真实数据的一部分
                if real_count > 0:
                    show_count = min(show_items, real_count)
                    print(f"    真实代码前{show_count}项: {codes[i, :show_count].tolist()}")
                    print(f"    真实时间前{show_count}项: {times[i, :show_count].tolist()}")

                # 显示部分填充数据
                if codes.shape[1] > real_count:
                    padding_start = real_count
                    padding_end = min(padding_start + show_items, codes.shape[1])
                    pad_show_count = padding_end - padding_start
                    if pad_show_count > 0:
                        print(f"    填充代码前{pad_show_count}项: {codes[i, padding_start:padding_end].tolist()}")
                        print(f"    填充时间前{pad_show_count}项: {times[i, padding_start:padding_end].tolist()}")

    # 实验室检测模态
    if 'lab' in batch:
        print("\n=== 实验室检测数据 ===")

        codes = batch['lab']['codes']
        times = batch['lab']['times']
        values = batch['lab']['values']
        masks = batch['lab']['mask']
        lengths = batch['lab']['lengths'] if 'lengths' in batch['lab'] else masks.sum(dim=1)

        print(f"  形状: {codes.shape}")
        print(f"  真实数据数量: {lengths[:show_samples].tolist()}")

        # 计算填充率
        padding_rates = [(1 - length.item() / codes.shape[1]) * 100 for length in lengths[:show_samples]]
        print(f"  填充率(%): {[f'{rate:.2f}%' for rate in padding_rates]}")

        # 显示部分样本的代码、时间和值
        for i in range(min(show_samples, batch_size)):
            real_count = lengths[i].item()
            print(f"  样本 {i} (真实数据: {real_count}, 填充率: {padding_rates[i]:.2f}%):")

            # 显示真实数据的一部分
            if real_count > 0:
                show_count = min(show_items, real_count)
                print(f"    真实代码前{show_count}项: {codes[i, :show_count].tolist()}")
                print(f"    真实时间前{show_count}项: {times[i, :show_count].tolist()}")
                print(f"    真实数值前{show_count}项: {values[i, :show_count].tolist()}")

            # 显示部分填充数据
            if codes.shape[1] > real_count:
                padding_start = real_count
                padding_end = min(padding_start + show_items, codes.shape[1])
                pad_show_count = padding_end - padding_start
                if pad_show_count > 0:
                    print(f"    填充代码前{pad_show_count}项: {codes[i, padding_start:padding_end].tolist()}")
                    print(f"    填充时间前{pad_show_count}项: {times[i, padding_start:padding_end].tolist()}")
                    print(f"    填充数值前{pad_show_count}项: {values[i, padding_start:padding_end].tolist()}")

    print("\n==== 批次检查完成 ====\n")



def patient_filter(visit_data, verbose=True):
    """
    过滤以visit为单位的数据，仅保留包含所有必需模态的visit记录

    必需的模态包括：
    - 诊断代码（diagnosis）
    - 药物代码（medication）
    - DRG代码（drg）
    - 出院摘要代码和嵌入（dis_codes, dis_embeddings）
    - 放射学报告代码和嵌入（rad_codes, rad_embeddings）
    - 实验室检测（lab_events）

    参数:
    visit_data: 以visit为单位的数据列表
    verbose: 是否显示详细信息

    返回:
    filtered_data: 过滤后的visit数据列表
    """
    total_visits = len(visit_data)
    if verbose:
        print(f"开始过滤，原始visit数量: {total_visits}")

    filtered_data = []

    for visit_record in visit_data:
        # 获取单个visit数据
        visit = visit_record['visit']

        # 初始化检查标志
        has_diagnosis = False
        has_medication = False
        has_drg = False
        has_dis_codes = False
        has_rad_codes = False
        has_dis_embeddings = False
        has_rad_embeddings = False
        has_lab_events = False

        # 检查诊断代码
        for diag_type in ['ccs_events', 'icd10_events', 'icd9_events', 'phecode_events']:
            if diag_type in visit and visit[diag_type]:
                has_diagnosis = True
                break

        # 检查药物代码
        if 'rxnorm_events' in visit and visit['rxnorm_events']:
            has_medication = True

        # 检查DRG代码
        for drg_type in ['drg_APR_events', 'drg_HCFA_events']:
            if drg_type in visit and visit[drg_type]:
                has_drg = True
                break

        # 检查出院摘要代码
        if 'dis_codes' in visit and visit['dis_codes']:
            has_dis_codes = True

        # 检查放射学报告代码
        if 'rad_codes' in visit and visit['rad_codes']:
            has_rad_codes = True

        # 检查出院摘要嵌入
        if 'dis_embeddings' in visit and visit['dis_embeddings']:
            has_dis_embeddings = True

        # 检查放射学报告嵌入
        if 'rad_embeddings' in visit and visit['rad_embeddings']:
            has_rad_embeddings = True

        # 检查实验室检测
        if 'lab_events' in visit and visit['lab_events']:
            has_lab_events = True

        # 如果该visit包含所有必需模态，则保留
        if (has_diagnosis and has_medication and has_drg and
                has_dis_codes and has_rad_codes and
                has_dis_embeddings and has_rad_embeddings and
                has_lab_events):
            filtered_data.append(visit_record)

    filtered_count = len(filtered_data)

    if verbose:
        print(f"过滤完成，结果:")
        print(f"原始visit数量: {total_visits}")
        print(f"过滤后visit数量: {filtered_count}")
        print(f"保留比例: {filtered_count / total_visits * 100:.2f}%")

    return filtered_data
def convert_to_visit_level_for_finetune(patient_data, verbose=True):
    """
    将以病人为单位的数据转换为以visit为单位的数据，排除last_visit

    参数:
    patient_data: 以病人为单位的原始数据列表，每个元素是一个患者字典
    verbose: 是否打印详细信息

    返回:
    visit_data: 以visit为单位的数据列表，每个元素是一个visit字典，排除last_visit
    """
    if verbose:
        print(f"开始将病人数据转换为visit数据（微调用），原始病人数量: {len(patient_data)}")

    visit_data = []
    total_visits = 0
    excluded_last_visits = 0

    for patient in patient_data:
        patient_id = patient.get('patient_id', 'unknown')
        demographics = patient.get('demographics', {})

        # 检查是否有visits数据
        if 'visits' not in patient or not patient['visits']:
            continue

        # 遍历患者的每次visit
        for visit_idx, visit in enumerate(patient['visits']):
            # 检查是否为last_visit
            if visit['is_last_visit']:
                print("tiaoguo")
                excluded_last_visits += 1
                continue  # 跳过last_visit

            # 创建新的visit记录
            visit_record = {
                'patient_id': patient_id,
                'demographics': demographics,
                'visit': visit,
                'visit_idx': visit_idx  # 记录这是患者的第几次就诊
            }
            visit_data.append(visit_record)
            total_visits += 1

    if verbose:
        print(f"转换完成，总共生成 {total_visits} 个visit记录")
        print(f"排除了 {excluded_last_visits} 个last_visit记录")
        print(f"平均每个患者的visit数量（排除last_visit后）: {total_visits / len(patient_data):.2f}")

    return visit_data
def check_patient_ids_and_times(batch):
    """检查批次中的病人ID和各模态的时间值"""
    print("\n==== 批次病人ID和时间检查 ====")

    # 获取病人ID
    if 'patient_id' in batch:
        patient_ids = batch['patient_id']
        for i, patient_id in enumerate(patient_ids):
            print(f"\n病人索引 {i}, ID: {patient_id}")

            # 检查各模态的时间值
            if 'diagnosis' in batch and batch['diagnosis']['times'] is not None:
                real_count = batch['diagnosis']['mask'][i].sum().item()
                if real_count > 0:
                    print(
                        f"  诊断时间范围: {batch['diagnosis']['times'][i, :real_count].min().item()} 到 {batch['diagnosis']['times'][i, :real_count].max().item()}")

            if 'medication' in batch and batch['medication']['times'] is not None:
                real_count = batch['medication']['mask'][i].sum().item()
                if real_count > 0:
                    print(
                        f"  药物时间范围: {batch['medication']['times'][i, :real_count].min().item()} 到 {batch['medication']['times'][i, :real_count].max().item()}")

            if 'drg' in batch and batch['drg']['times'] is not None:
                real_count = batch['drg']['mask'][i].sum().item()
                if real_count > 0:
                    print(
                        f"  DRG时间范围: {batch['drg']['times'][i, :real_count].min().item()} 到 {batch['drg']['times'][i, :real_count].max().item()}")

            if 'discharge_summary' in batch and batch['discharge_summary']['times'] is not None:
                real_count = batch['discharge_summary']['mask'][i].sum().item()
                if real_count > 0:
                    print(
                        f"  出院摘要时间范围: {batch['discharge_summary']['times'][i, :real_count].min().item()} 到 {batch['discharge_summary']['times'][i, :real_count].max().item()}")

            if 'radiology_report' in batch and batch['radiology_report']['times'] is not None:
                real_count = batch['radiology_report']['mask'][i].sum().item()
                if real_count > 0:
                    print(
                        f"  放射学报告时间范围: {batch['radiology_report']['times'][i, :real_count].min().item()} 到 {batch['radiology_report']['times'][i, :real_count].max().item()}")

            if 'discharge_codes' in batch and batch['discharge_codes']['times'] is not None:
                real_count = batch['discharge_codes']['mask'][i].sum().item()
                if real_count > 0:
                    print(
                        f"  出院代码时间范围: {batch['discharge_codes']['times'][i, :real_count].min().item()} 到 {batch['discharge_codes']['times'][i, :real_count].max().item()}")

            if 'radiology_codes' in batch and batch['radiology_codes']['times'] is not None:
                real_count = batch['radiology_codes']['mask'][i].sum().item()
                if real_count > 0:
                    print(
                        f"  放射学代码时间范围: {batch['radiology_codes']['times'][i, :real_count].min().item()} 到 {batch['radiology_codes']['times'][i, :real_count].max().item()}")

            if 'lab' in batch and batch['lab']['times'] is not None:
                real_count = batch['lab']['mask'][i].sum().item()
                if real_count > 0:
                    print(
                        f"  实验室检测时间范围: {batch['lab']['times'][i, :real_count].min().item()} 到 {batch['lab']['times'][i, :real_count].max().item()}")

            # 只显示前几个病人
            if i >= 2:
                break

    print("\n==== 检查完成 ====\n")

