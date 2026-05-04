#!/bin/bash

export PYTHONPATH="/raid/home/xukai/FRATTVAE/scripts:$PYTHONPATH"
export DGLBACKEND="pytorch"

# source your environment

data_dir="/raid/home/xukai/git_data/FRATTVAE/"  # 文件所在目录

# 遍历目录中的所有 CSV 文件并标准化
for file_path in "$data_dir"*.csv; do
    log_file="prepare_$(basename "$file_path" .csv).log"  # 每个文件对应一个独立日志
    echo "Standardizing: $file_path" | tee -a $log_file  # 输出到控制台和日志文件
    echo "Start time: $(date)" | tee -a $log_file  # 记录开始时间
    python utils/standardize_smiles.py "$file_path" --n_jobs 16 >> $log_file 2>&1  # 将标准化日志记录到文件
    echo "End time: $(date)" | tee -a $log_file  # 记录结束时间
    echo "----------------------------------------" | tee -a $log_file  # 分隔符
done