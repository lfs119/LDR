# success_rate.py
import csv
import json
import os
from rdkit import Chem
from rdkit.Chem import QED
from moses import metrics

from tdc import Oracle      # 内置 GSK3β/JNK3 二分类器
import torch
# import shutil
# shutil.rmtree(os.path.expanduser('~/.tdc/cache/'), ignore_errors=True)

# # 1. 加载靶点预测器（GPU 可用）
# gsk3b = Oracle('GSK3β')
# jnk3  = Oracle('JNK3')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型（显式指定设备）
gsk3b = Oracle('GSK3β', device=device, force_download=True)
jnk3  = Oracle('JNK3', device=device, force_download=True)

# 2. 单分子四闸门判定
def success_one(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    g = gsk3b(smiles)          # 概率 ∈[0,1]
    j = jnk3(smiles)
    q = QED.qed(mol)
    s = metrics.SA(mol)  # 原始 SA 分数
    return g >= 0.5 and j >= 0.5 and q >= 0.6 and s <= 4

# 3. 批量成功率
def success_rate(smiles_list):
    ok = sum(success_one(smi) for smi in smiles_list)
    return ok / len(smiles_list)

# 4. 快速测试
if __name__ == '__main__':
    # base_path = '/home/wangqh/xk/FRATTVAE/runs/frattvae_pro_20251010_102809/'
    # json_path = base_path + 'topk.json'
    # csv_output_path = base_path + 'topk_properties.csv'
    # with open(json_path, 'r') as f:
    #     data = json.load(f)
    # sum_list= []
    # for item in data:
    #         smiles = item[1] 
    #         sum_list.append(success_one(smiles))
    # rate = sum(sum_list) / len(sum_list)

   
    demo = ['Cc1ccccc1', 'CC(=O)Nc1ccc(O)cc1', 'CCCN1CCC(O)CC1']  # 随意例子
    rate = success_rate(demo)
    print(f'成功率 = {rate:.1%}')