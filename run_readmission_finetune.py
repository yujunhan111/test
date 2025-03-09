import os
import sys


def set_gpu():
    """设置使用的GPU"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一张GPU


def main():
    # 设置GPU
    set_gpu()

    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 添加finetuing目录到Python路径
    sys.path.append(os.path.join(current_dir, 'finetuing'))

    # 设置微调命令行参数
    sys.argv = [
        sys.argv[0],
        "--model_name", "FusionBert_coAtt_residual_readm_predict",
        "--data_dir", os.path.join(current_dir, "preproced_data"),
        "--output_dir", os.path.join(current_dir, "finetune_results", "readmission"),
        "--pretrain_dir", os.path.join(current_dir, "output", "multi_modal_coAtt_residual_pretrain"),
        "--clinical_bert_dir", os.path.join(current_dir, "clinical_bert_dir"),  # 修改为与预训练相同的路径
        "--graph",
        "--do_train",
        "--data_name", "binarylabel",
        "--predict_task", "readm",
        "--model_choice", "fusion_bin",
        "--train_batch_size", "16",
        "--eval_batch_size", "16",
        "--learning_rate", "2e-5",
        "--num_train_epochs", "20",
        "--use_pretrain",
        "--max_visit_len", "10",
        "--seed", "2023"
    ]

    # 导入微调脚本
    from finetuing.fusionBert_predict import main as finetune_main

    # 运行微调
    finetune_main()


if __name__ == "__main__":
    main()