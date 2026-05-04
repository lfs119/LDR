import pandas as pd
import os, sys
# 1. 读取 CSV 文件（假设列名为 'SMILES'）
csv_path = '/raid/home/xukai/FRATTVAE/data/coconut2_202509_r1r10w1100w900_standardized.csv'
csv_path = '/raid/home/xukai/FRATTVAE/data/ZINC_JTVAE_standardized.csv'
csv_path = '/raid/home/xukai/FRATTVAE/data/chembl_36_20251023_r1r10w1100w900_standardized_f_dyn_oral.csv'

df = pd.read_csv(csv_path)
csv_name = csv_path.split('/')[-1].split('.')[0]
csv_dir = os.path.dirname(csv_path)
# 2. 确保 SMILES 列存在
if "SMILES" not in df.columns:
    raise ValueError("CSV 文件中没有 'SMILES' 列！")

# 3. 去除空值和重复项（可选）
smiles_list = df["SMILES"].dropna().unique()

output_path = os.path.join(csv_dir, f"{csv_name}.smi")
# 4. 写入 .smi 文件（每行一个 SMILES）
with open(output_path, "w") as f:
    for smi in smiles_list:
        f.write(str(smi).strip() + "\n")

print(f"已成功写入 {len(smiles_list)} 个 SMILES 到 output.smi")