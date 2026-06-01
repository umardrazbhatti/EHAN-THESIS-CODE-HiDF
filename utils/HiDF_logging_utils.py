"""
utils/logging_utils.py — TensorBoard writer + CSV fallback logger.
"""

import csv
import os
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.writer   = SummaryWriter(log_dir)
        csv_path      = os.path.join(log_dir, "logs.csv")
        self.csv_file = open(csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["step", "tag", "value"])

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        for k, v in tag_scalar_dict.items():
            self.writer.add_scalar(f"{main_tag}/{k}", v, step)
            self.csv_writer.writerow([step, f"{main_tag}/{k}", v])
        self.csv_file.flush()

    def close(self):
        self.writer.close()
        self.csv_file.close()
