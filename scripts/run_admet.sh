#!/bin/bash

# 设置基础参数（这些在所有命令中保持不变）
BASE_PATH="/raid/home/xukai/git_data/FRATTVAE"
ADMET_KEYS="/raid/home/xukai/FRATTVAE/results/CNS_SMILES_standardized_struct_1020/input_data/admet_par.json"
PROJECT="oral_cns"

# 要处理的 CSV 文件列表
CSV_FILES=(
    "GuacaMol_standardized.csv"
    "MOSES_standardized.csv"
    "Polymer_standardized.csv"
    # "SuparNatural_over500_standardized.csv"
)

# 遍历每个 CSV 文件并执行命令
for csv in "${CSV_FILES[@]}"; do
    echo "Processing $csv ..."
    CUDA_VISIBLE_DEVICES=1 python models/add_admet_to_csv.py \
        --base_path "$BASE_PATH" \
        --csv_name "$csv" \
        --admet_keys "$ADMET_KEYS" \
        --project "$PROJECT"
done    

echo "All done!"