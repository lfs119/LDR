#!/bin/bash

export PYTHONPATH="/home/xukai/xk/FRATTVAE/scripts:$PYTHONPATH"
export DGLBACKEND="pytorch"
export PYTHONWARNINGS="ignore"

# 基础路径配置
base_path="/home/wangqh/xk/FRATTVAE"
results_path="${base_path}/results/coconut2_202509_r1r10w1100w900_standardized_struct_0915"
yml_file="${results_path}/input_data/params.yml"
timestamp="frattvae_pro_20251013_195244"
device="2"  # CUDA设备编号
n_runs=3   # 固定参数

# 定义不同checkpoint的配置
declare -A checkpoints=(
    ["350"]="policy_350.pt"
    ["300"]="policy_300.pt"
    ["250"]="policy_250.pt"
)

# 遍历执行
for key in "${!checkpoints[@]}"; do
    ckpt_file="${base_path}/runs/${timestamp}/${checkpoints[$key]}"
    output_file="${results_path}/${timestamp}_scaffold__p${key}.txt"
    
    echo "Processing ${ckpt_file} -> ${output_file}"
    
    CUDA_VISIBLE_DEVICES=${device} python test_rl.py \
        "${yml_file}" \
        --ckpt "${ckpt_file}" \
        --N ${n_runs} \
        > "${output_file}"
done

