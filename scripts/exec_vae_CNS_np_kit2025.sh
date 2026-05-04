#!/bin/bash

export PYTHONPATH="/home/wangqh/xk/FRATTVAE/scripts:$PYTHONPATH"
export DGLBACKEND="pytorch"
export PYTHONWARNINGS="ignore"

path="/raid/home/xukai/FRATTVAE/results/CNS_SMILES_standardized_struct_1021"
ymlFile=$path'/input_data/params.yml'
load_epoch=0

# python3 preprocessing.py ${ymlFile} --n_jobs 16 --biosynfoni >> $path'/preprocess_CNS_biosynfoni.log' &
python3  -W ignore train.py ${ymlFile} --gpus 1 2 3    --n_jobs 24 --save_interval 200 --load_epoch $load_epoch --valid  > $path'/train'$load_epoch'.log' &&
python3  -W ignore test.py ${ymlFile} --N 5 --k 10000 --gpu 2 --n_jobs 16 > $path'/test.log' &

# nohup python test.py ${ymlFile} --N 5 --k 10000 --gpu 7 --n_jobs 24 --load_epoch $load_epoch  > $path'/test'$load_epoch'.log' &
# nohup python test.py ${ymlFile} --N 5 --k 10000 --gpu 7 --n_jobs 24 --max_nfrags 12 --load_epoch $load_epoch --gen > $path'/generate'$load_epoch'.log' &

