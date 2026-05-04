#!/bin/bash

export PYTHONPATH="/raid/home/xukai/FRATTVAE/scripts:$PYTHONPATH"
export DGLBACKEND="pytorch"
export PYTHONWARNINGS="ignore"

path="/raid/home/xukai/FRATTVAE/results/ZINC_JTVAE_standardized_struct_1015"
path="/raid/home/xukai/FRATTVAE/results/GuacaMol_standardized_struct_1107"   # admet_data_filtered_molecules_lrrk2_0_struct_1118
path="/raid/home/xukai/FRATTVAE/results/admet_data_filtered_molecules_lrrk2_0.5_struct_1118"
ymlFile=$path'/input_data/params.yml'
load_epoch=-1

# python3 preprocessing.py ${ymlFile} --n_jobs 16 --biosynfoni >> $path'/preprocess_biosynfoni_lrrk2_0.5.log' &
python3  -W ignore train.py ${ymlFile} --gpus 3 --n_jobs 24 --save_interval 100 --load_epoch $load_epoch --valid  > $path'/train'$load_epoch'.log' &&
python3  -W ignore test.py ${ymlFile} --N 3 --k 10000 --gpu 1 --n_jobs 24 --gen > $path'/test.log' &

# nohup python test.py ${ymlFile} --N 5 --k 10000 --gpu 7 --n_jobs 24 --load_epoch $load_epoch  > $path'/test'$load_epoch'.log' &
# nohup python test.py ${ymlFile} --N 5 --k 10000 --gpu 7 --n_jobs 24 --max_nfrags 12 --load_epoch $load_epoch --gen > $path'/generate'$load_epoch'.log' &
