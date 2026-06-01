"""
run_full_pipeline.py — Entry point for Kaggle / local execution.

Usage (Kaggle, fresh run):
    !python run_full_pipeline.py \\
        --data_root /kaggle/input/.../ffpp_data \\
        --dataset_name ff++ \\
        --epochs 10 \\
        --batch_size 4 \\
        --eval_after_train

Usage (Kaggle, resume mid-run, skip eval):
    !python run_full_pipeline.py \\
        --data_root /kaggle/input/.../ffpp_data \\
        --resume_checkpoint /kaggle/working/outputs/last_checkpoint.pth \\
        --epochs 5 \\
        --skip_eval

Usage (local synthetic smoke test):
    python run_full_pipeline.py --dataset_name synthetic --epochs 2 --batch_size 2
"""

import os
from HiDF_config import EAHNConfig, parse_args
from scripts.HiDF_train_real import main as train_main
from scripts.HiDF_dashboard import show_dashboard


def main():
    args   = parse_args()
    config = EAHNConfig.from_args(args)
    os.makedirs(config.output_dir, exist_ok=True)
    print(f"Output directory:        {config.output_dir}")
    print(f"Device:                  {config.device}")
    print(f"Dataset:                 {config.dataset_name}")
    print(f"Active manipulation:     {config.active_manipulation}")
    print(f"resume_checkpoint:       {config.resume_checkpoint or '(none)'}")
    print(f"skip_eval:               {config.skip_eval}")
    print(f"celebdf_eval:            {config.celebdf_eval}")
    print(f"celebdf_root:            {config.celebdf_root or '(none)'}")
    print(f"save_last_checkpoint:    {config.save_last_checkpoint}")
    print(f"explanation_suite:       {config.explanation_suite}")

    if config.skip_eval:
        config.eval_after_train = False
        print("[run_full_pipeline] --skip_eval set, skipping evaluation.")

    train_main(config)
    show_dashboard(config.output_dir)
    print("Full pipeline completed. Outputs in", config.output_dir)


if __name__ == "__main__":
    main()
