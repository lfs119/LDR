#!/bin/bash

export PYTHONPATH="/home/xukai_cluster/FRATTVAE_800/scripts:$PYTHONPATH"
export DGLBACKEND="pytorch"


# data_path="/raid/home/xukai/git_data/FRATTVAE/ZINC_JTVAE_standardized.csv"
# data_path="/raid/home/xukai/FRATTVAE/data/ChemBERT_10m_standardized.csv"
# data_path="/raid/home/xukai/FRATTVAE/data/CNS_SMILES_standardized.csv"
# data_path="/home/wangqh/xk/FRATTVAE/data/coconut2_202509_r1r10w1100w900_standardized.csv"
# data_path="/raid/home/xukai/FRATTVAE/data/chembl_36_20251023_r1r10w1100w900_standardized.csv"
# data_path="/raid/home/xukai/FRATTVAE/data/chembl_36_20251023_r1r10w1100w900_standardized_f_dyn_oral.csv"
# data_path="/raid/home/xukai/FRATTVAE/data/chembl_36_20251023_r1r10w1100w900_standardized_f_dyn_cns.csv"
# data_path="/raid/home/xukai/git_data/FRATTVAE/GuacaMol_standardized.csv"
# data_path="/raid/home/xukai/FRATTVAE/data/admet_data/admet_data_filtered_molecules_lrrk2_0.5_split_0.90_0.05_0.05.csv"
# data_path="/raid/home/xukai/FRATTVAE/data/admet_data/admet_data_filtered_molecules_hpk1_0.5_split_0.90_0.05_0.05.csv"
# data_path="/raid/home/xukai/FRATTVAE/data/admet_data/admet_data_filtered_molecules_hpk1_0.5_0.7_split_0.90_0.05_0.05.csv"

data_path="/home/xukai_cluster/FRATTVAE_800/data/fluo4.3/FluoDB_smiles_standardized.csv"
echo date >> prepare_rdkit_FluoDB_smiles.log

#  生成训练配置

# basic     12288
python preparation.py $data_path \
                      --seed 0 \
                      --maxLength 64 \
                      --maxDegree 16 \
                      --minSize 4 \
                      --epoch 1000 \
                      --batch_size 1800 \
                      --lr 0.0001 \
                      --kl_w 0.0005 \
                      --l_w 6.0 >> FluoDB.log    # 10.08      2.0->6.0
                     

# # kl-annealing
# python preparation.py $data_path \
#                       --seed 0 \
#                       --maxLength 32 \
#                       --maxDegree 16 \
#                       --minSize 1 \
#                       --epoch 1000 \sh 
#                       --batch_size 512 \
#                       --lr 0.0001 \
#                       --kl_w 0.0005 \
#                       --anneal_epoch 500:10 \
#                       --l_w 2.0 >> prepare.log

# # conditional
# python preparation.py $data_path \
#                       --seed 0 \
#                       --maxLength 32 \
#                       --maxDegree 16 \
#                       --minSize 1 \
#                       --epoch 1000 \
#                       --batch_size 1024 \
#                       --condition MW:1 \
#                       --condition logP:1 \
#                       --condition QED:1 \
#                       --condition SA:1 \
#                       --condition NP:1 \
#                       --condition TPSA:1 \
#                       --lr 0.0001 \
#                       --kl_w 0.0005 \
#                       --l_w 2.0 >> prepare.log