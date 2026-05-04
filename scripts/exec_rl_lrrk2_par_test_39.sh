#!/bin/bash

export PYTHONPATH="/home/xukai/xk/FRATTVAE/scripts:$PYTHONPATH"
export DGLBACKEND="pytorch"
export PYTHONWARNINGS="ignore"

# 参数检查（必须4个参数）
if [ $# -ne 4 ]; then
    echo "Usage: $0 base_path timestamp device n_runs"
    echo "Example: $0 /home/wangqh/xk/FRATTVAE 20251013_124714 2 3"
    exit 1
fi

# 参数解析
base_path=$1
timestamp=$2
device=$3
n_runs=${4:-3}  # 默认3次运行
num=10000

results_path="${base_path}/results/CNS_SMILES_standardized_struct_1021"
yml_file="${results_path}/input_data/params.yml"
checkpoint_dir="${base_path}/runs/${timestamp}"
echo "Using checkpoint directory: $checkpoint_dir"

# 路径验证
[[ -d "$base_path" ]] || { echo "Error: Base directory not found!"; exit 1; }
[[ -d "$results_path" ]] || { echo "Error: Results directory missing!"; exit 1; }
[[ -d "$checkpoint_dir" ]] || { echo "Error: Checkpoint directory not found!"; exit 1; }

# 检查点配置
declare -A checkpoints=(
    # [350]="policy_350.pt"
    # [300]="policy_300.pt"
    [250]="policy_250.pt"
    [200]="policy_200.pt"
    # [150]="policy_150.pt"
)

# 执行循环
for key in "${!checkpoints[@]}"; do
    ckpt_file="${checkpoint_dir}/${checkpoints[$key]}"
    output_file="${results_path}/${timestamp}_dat_p${key}.txt"
    
    if [[ ! -f "$ckpt_file" ]]; then
        echo "Warning: Checkpoint file not found -> $ckpt_file (skipping)"
        continue
    fi

    echo "Processing $ckpt_file -> $output_file"
    CUDA_VISIBLE_DEVICES=$device python test_rl.py \
        "$yml_file" --ckpt "$ckpt_file" --N "$n_runs" --k "$num"> "$output_file"
done