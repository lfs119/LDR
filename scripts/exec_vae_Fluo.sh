#!/bin/bash

export PYTHONPATH="/home/xukai_cluster/FRATTVAE_800/scripts:$PYTHONPATH"
export DGLBACKEND="pytorch"
export PYTHONWARNINGS="ignore"

# path="/home/admin/xk/FRATTVAE/results/ZINC_JTVAE_standardized_struct"
path="/home/xukai_cluster/FRATTVAE_800/results/FluoDB_smiles_standardized_struct_0407"
ymlFile=$path'/input_data/params.yml'
load_epoch=0

# python3 preprocessing.py ${ymlFile} --n_jobs 16 >> $path'/preprocess_fluo_0407.log' &
python3  -W ignore train.py ${ymlFile} --gpus 5 6 7  --n_jobs 24 --save_interval 50 --load_epoch $load_epoch --valid  > $path'/train'$load_epoch'.log' &&
python3  -W ignore test.py ${ymlFile} --N 2 --k 10000 --gpu 7 --n_jobs 24 > $path'/test.log' &

# nohup python test.py ${ymlFile} --N 5 --k 10000 --gpu 7 --n_jobs 24 --load_epoch $load_epoch  > $path'/test'$load_epoch'.log' &
# nohup python test.py ${ymlFile} --N 5 --k 10000 --gpu 7 --n_jobs 24 --max_nfrags 12 --load_epoch $load_epoch --gen > $path'/generate'$load_epoch'.log' &
