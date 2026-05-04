#!/bin/bash

export PYTHONPATH="/raid/home/xukai/FRATTVAE/scripts:$PYTHONPATH"
export DGLBACKEND="pytorch"

# source yourenviroment

# data_path="/raid/home/xukai/git_data/FRATTVAE/ChemBERT_10m_standardized.csv"
# data_path="/home/wangqh/xk/FRATTVAE/data/coconut2_202509_r1r10w1100w900.csv"
data_path="/raid/home/xukai/FRATTVAE/data/chembl_36_20251023_r1r10w1100w900.csv"
data_path="/raid/home/xukai/git_data/FRATTVAE/SuparNatural_over500_standardized.csv"   # SuparNatural_over500_standardized.csv    MOSES_standardized
data_path="/home/xukai_cluster/FRATTVAE_800/data/fluo4.3/FluoDB_smiles.csv"

# Only the first time
python utils/standardize_smiles.py $data_path --n_jobs 16 >> prepare_standardize.log
