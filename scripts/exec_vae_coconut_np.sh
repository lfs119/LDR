#!/bin/bash

export PYTHONPATH="/home/wangqh/xk/FRATTVAE/scripts:$PYTHONPATH"
export DGLBACKEND="pytorch"
export PYTHONWARNINGS="ignore"

path="/home/wangqh/xk/FRATTVAE/results/coconut2_202509_r1r10w1100w900_standardized_struct_1008"
ymlFile=$path'/input_data/params.yml'
load_epoch=0

# python3 preprocessing.py ${ymlFile} --n_jobs 16 >> $path'/preprocess.log' &
python3  -W ignore train.py ${ymlFile} --gpus 2 3 4   --n_jobs 12 --save_interval 200 --load_epoch $load_epoch --valid  > $path'/train'$load_epoch'.log' &&
python3  -W ignore test.py ${ymlFile} --N 5 --k 10000 --gpu 2 --n_jobs 24 > $path'/test.log' &

# nohup python test.py ${ymlFile} --N 5 --k 10000 --gpu 7 --n_jobs 24 --load_epoch $load_epoch  > $path'/test'$load_epoch'.log' &
# nohup python test.py ${ymlFile} --N 5 --k 10000 --gpu 7 --n_jobs 24 --max_nfrags 12 --load_epoch $load_epoch --gen > $path'/generate'$load_epoch'.log' &
