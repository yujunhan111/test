from disease_codes import DISEASE_CODES
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
from tqdm import tqdm


def evaluate_model_disease(model, val_loader, device):
    """
    验证模型并收集每种疾病的预测结果
    """
    model.eval()
    disease_predictions = {}

    # 初始化疾病预测收集器
    for disease_name in model.disease_names:
        disease_predictions[disease_name] = {
            'predictions': [],
            'labels': []
        }

    # 添加进度条
    total_samples = len(val_loader.dataset)
    progress_bar = tqdm(total=total_samples, desc="Validating", unit="patient")

    with torch.no_grad():
        for batch in val_loader:
            # 将数据转移到正确的设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # 获取预测
            prediction_results = model(batch, is_training=False)

            # 处理预测结果
            for name, pred_data in prediction_results.items():
                # 确保name在disease_predictions中
                if name in disease_predictions:
                    disease_predictions[name]['predictions'].extend(pred_data['predictions'])
                    disease_predictions[name]['labels'].extend(pred_data['labels'])

            # 更新进度条
            progress_bar.update(batch['demographic'].size(0))

    progress_bar.close()
    return disease_predictions


def evaluate_epoch(model, val_loader, device):
    """
    评估一个epoch并返回所有指标
    """
    # 获取所有预测结果
    disease_predictions = evaluate_model_disease(model, val_loader, device)

    # 计算每个疾病的指标
    disease_metrics = {}
    avg_metrics = {'auroc': 0, 'auprc': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    valid_diseases = 0

    # 处理所有预测结果
    for name in model.disease_names:
        if name in disease_predictions and disease_predictions[name]['predictions']:
            predictions = torch.stack(disease_predictions[name]['predictions']).cpu().numpy() if isinstance(
                disease_predictions[name]['predictions'][0], torch.Tensor) else np.array(
                disease_predictions[name]['predictions'])
            labels = np.array(disease_predictions[name]['labels'])

            metrics = calculate_metrics(predictions, labels)

            # 只有当指标不是nan时才计入平均值
            if not np.isnan(metrics['auroc']):
                disease_metrics[name] = metrics
                for metric in avg_metrics:
                    avg_metrics[metric] += metrics[metric]
                valid_diseases += 1
            else:
                print(f"Skipping metrics for {name} due to insufficient class distribution")

    # 计算疾病预测的平均指标
    if valid_diseases > 0:
        for metric in avg_metrics:
            avg_metrics[metric] /= valid_diseases

    return {
        'disease_metrics': disease_metrics,
        'average_metrics': avg_metrics
    }


def calculate_metrics(predictions, labels):
    """
    计算评估指标并统计样本数量

    Args:
        predictions: 预测概率列表
        labels: 真实标签列表 (0.95表示正例, 0.05表示负例)

    Returns:
        metrics字典包含AUROC、AUPRC、最佳阈值、最佳Precision、最佳Recall、最佳F1以及样本统计信息
    """
    predictions = np.array(predictions)
    binary_labels = np.array([1 if l > 0.5 else 0 for l in labels])

    # 统计样本数量
    total_samples = len(binary_labels)
    positive_samples = np.sum(binary_labels)
    negative_samples = total_samples - positive_samples

    # 检查是否只有一个类别
    unique_labels = np.unique(binary_labels)
    if len(unique_labels) < 2:
        return {
            'auroc': float('nan'),
            'auprc': float('nan'),
            'precision': float('nan'),
            'recall': float('nan'),
            'f1': float('nan'),
            'total_samples': total_samples,
            'positive_samples': positive_samples,
            'negative_samples': negative_samples,
            'positive_ratio': float(positive_samples) / total_samples if total_samples > 0 else 0
        }

    try:
        # 计算AUROC
        auroc = roc_auc_score(binary_labels, predictions)

        # 计算Precision-Recall曲线
        precision, recall, thresholds = precision_recall_curve(binary_labels, predictions)
        auprc = auc(recall, precision)

        # 计算F1分数
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_index = np.argmax(f1_scores)  # 最大F1分数的索引

        # 获取最佳阈值及相关指标
        best_threshold = thresholds[best_index] if best_index < len(thresholds) else 1.0
        best_precision = precision[best_index]
        best_recall = recall[best_index]
        best_f1 = f1_scores[best_index]

        return {
            'auroc': auroc,
            'auprc': auprc,
            'precision': best_precision,
            'recall': best_recall,
            'f1': best_f1,
            'total_samples': total_samples,
            'positive_samples': positive_samples,
            'negative_samples': negative_samples,
            'positive_ratio': float(positive_samples) / total_samples
        }
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {
            'auroc': float('nan'),
            'auprc': float('nan'),
            'precision': float('nan'),
            'recall': float('nan'),
            'f1': float('nan'),
            'total_samples': total_samples,
            'positive_samples': positive_samples,
            'negative_samples': negative_samples,
            'positive_ratio': float(positive_samples) / total_samples if total_samples > 0 else 0
        }


def print_metrics_summary(val_metrics):
    """
    打印评估指标和样本统计的详细摘要
    """
    print("\nValidation Results Summary:")

    # 打印平均指标
    print("\nAverage Metrics Across All Diseases:")
    avg_metrics = val_metrics['average_metrics']
    print(f"AUROC: {avg_metrics['auroc']:.4f}")
    print(f"AUPRC: {avg_metrics['auprc']:.4f}")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f}")
    print(f"F1: {avg_metrics['f1']:.4f}")

    # 打印每个疾病的详细指标
    print("\nPer-disease Metrics:")
    disease_metrics = val_metrics['disease_metrics']

    # 创建表格格式的输出
    headers = ["Disease", "AUROC", "AUPRC", "F1", "Precision", "Recall", "Total", "Pos", "Pos%"]
    format_str = "{:<30} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8} {:<8} {:<8.2%}"

    print("\n" + " ".join(f"{h:<8}" for h in headers))
    print("-" * 100)

    # 只打印疾病预测结果
    for disease, metrics in disease_metrics.items():
        if not np.isnan(metrics['auroc']):
            print(format_str.format(
                disease,
                metrics['auroc'],
                metrics['auprc'],
                metrics['f1'],
                metrics['precision'],
                metrics['recall'],
                metrics['total_samples'],
                metrics['positive_samples'],
                metrics['positive_ratio']
            ))
        else:
            print(f"{disease:<30} Insufficient data for metrics calculation")