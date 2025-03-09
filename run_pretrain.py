import os
import sys
from src.get_options import args_parser


def set_gpu():
    """设置使用的GPU"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一张GPU


def main():
    # 设置GPU
    set_gpu()

    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 设置参数
    sys.argv = [
        sys.argv[0],
        "--model_name", "multi_modal_coAtt_residual_pretrain",
        "--num_train_epochs", "200",
        "--do_train",
        "--graph",
        "--clinical_bert_dir", os.path.join(current_dir, "clinical_bert_dir"),
        "--data_dir", os.path.join(current_dir, "preproced_data"),
        "--output_dir", os.path.join(current_dir, "output"),
        "--train_batch_size", "32",
        "--learning_rate", "2e-5",
    ]

    # 解析参数
    args = args_parser()

    # 导入主程序
    from main import main as main_program

    # 运行主程序
    main_program()


if __name__ == "__main__":
    main()