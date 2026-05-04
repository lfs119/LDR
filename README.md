# LDR-MD

This repository contains training and generation code for LDR-MD. a closed-loop framework that integrates fragment-aware latent diffusion, Pareto-guided multi-objective reinforcement learning, and molecular dynamics-based binding validation for end-to-end molecular design with experimental validation.

## Requirements
* Python==3.10.8
* Scipy==1.11.3
* Pytorch>=1.12.1 (We only testd 1.12.1 and 2.0.1)
* DGL>=1.1.1
* RDkit==2023.2.1
* molvs (https://github.com/mcs07/MolVS)
* moses (https://github.com/molecularsets/moses)
* guacamol (https://github.com/BenevolentAI/guacamol)

To install these packages, follow the respective instructions.

## Data availability
* ZINC250K: https://github.com/kamikaze0923/jtvae/tree/master/data/zinc
* MOSES: https://github.com/molecularsets/moses/tree/master/data
* Chembl_36: https://doi.org/10.6019/CHEMBL.database.36
* Coconut2: https://coconut.s3.uni-jena.de/prod/downloads
* CNS: https://github.com/xueww/CNSGT/tree/main/data

And, you can download several standardized datasets [here](https://doi.org/10.5281/zenodo.19969289) (ZINC250K, Chembl_36, coconut2, cns).

## Quick Start
Essential packages can be installed via `pip`, but the version of CUDA is up to you (Default: 11.3). 
Please execute `enviroment.sh` in your python virtual enviroment.
```
sh enviroment.sh
pip install QEPPIcommunity
pip install qrci
pip install meeko
```
If you use Docker, you can use the Dockerfile to build your environment.
```
docker build . --network=host -t <IMAGE NAME>
docker run -itd --runtime=nvidia --shn-size 32g -t <CONTAINER NAME> <IMAGE ID>
```

## 0. Preparation
### (0.0.) 　Standardize SMILES
To canonicalize and sanitize SMILES, run `exec_standardize.sh` only once. your data must be in csv format and have a column named 'SMILES'. If there is not a column called 'test' in your data, it will be split into train/valid/test data sets (0: train, 1: test, -1: valid). The standardized data is saved as `*_standardized.csv`.

exec_standardize.sh:
```
#!/bin/bash

export PYTHONPATH="/yourdirectory/scripts:$PYTHONPATH"
export DGLBACKEND="pytorch"

source yourenviroment
data_path="/yourdirectory/data/example.csv"

python utils/standardize_smiles.py $data_path --n_jobs 24 >> prepare.log
```
* `--n_jobs`: Number of cpu workers.

Please change 'yourdirectory' and 'yourenviroment' to the correct paths.

### 0.1. 　Setting Hyperparameters and Directory to save results
exec_prepare.sh:
```
python preparation.py "/yourdirectory/data/example_standardized.csv" \
                      --seed 0 \
                      --maxLength 32 \
                      --maxDegree 16 \
                      --minSize 1 \
                      --epoch 1000 \
                      --batch_size 2048 \
                      --lr 0.0001 \
                      --kl_w 0.0005 \
                      --l_w 2.0 >> prepare.log 
```
After execution, `savedir` named `dataname_{taskname}_{MMDD}` in `/yourdirectory/results/.`, and `params.yml` which is hyperparameters list in `/savedir/input_data/.` are created.

Please adjust hyperparameters (batch_size, maxLength and so on) to match your datasets and GPU capacities.
For molecules with molecular weights greater than 500, it is recommended that maxLength be 32 or 64.

## 1. Precedure of Training and Generation (VAE + Diffusion)
Please refer to `exec_vae.sh`.

### 1.1. Preprocessing
```
path="/yourdirectory/results/examples_standardized_struct_{MMDD}"
ymlFile=$path'/input_data/params.yml'
python preprocessing.py ${ymlfile} --njobs 24 >> $path'/preprocess.log'
```
* `--ymlfile`: the path of `params.yml`.
* `--n_jobs`: Number of cpu workers.

After execution, `fragments.csv` and `dataset.pkl` are created in `/savedir/input_data/.`

### 1.2. Training VAE
```
python train.py ${ymlFile} --gpus 0 1 2 3 --n_jobs 24 --load_epoch $load_epoch --valid > $path'/train'$load_epoch'.log'
```
* `--gpus`: IDs of GPU. If multiple GPUs are given, they are used for DDP training.
* `--n_jobs`: Number of cpu workers.
* `--load_epoch`: load `$load_epoch`-epoch trained model. Use to resume learning from any epoch.
* `--valid`: To Validate or Not to Validate.

After execution, the model checkepoint is saved as `model_best.pth` in `/savedir/model/.`

### 1.3. Generation VAE
Caluculate MOSES+GuacaMol metrics.
```
python test.py ${ymlFile} --gpu 0 --k 10000 --N 5 --n_jobs 24 > $path'/test.log'
```
* `--gpu`: ID of GPU. multi GPUs are not supported.
* `--k`: Number of moldecules generated.
* `--N`: Iteration number of generation.
* `--n_jobs`: Number of cpu workers.
* `--gen`: Set if you only want Generation.

After execution, the results of reconstruction and generation are saved in `/savedir/test/.` and `/savedir/generate/.` respectively.

### 1.4. Training Diffusion
```
CUDA_VISIBLE_DEVICES="1,2,3" accelerate launch --main_process_port 12903 --multi_gpu --mixed_precision fp16  train_diff.py --model diff_unet --cfgdir  path/diff_unet.yaml   --tgtdir   tath/models_diff  --cfgvae path/params.yml
```
* `--main_process_port`: port number
* `--cfgdir`: diffusion config file
* `--tgtdir`: svaedir diffusion wegihts
* `--cfgvae` : vae config file

### 1.5. Generation Diffusion
```
CUDA_VISIBLE_DEVICES="0" accelerate launch --main_process_port 12904 train_diff.py  --model generate   --cfgdir  path/diff_unet.yaml    --cfgvae path/params.yml --gen_unet_path  path/diffusion.pth --N 5 --gen_vae_path path/vae.pth
```
* `--main_process_port`: port number
* `--cfgdir`: diffusion config file
* `--gen_unet_path`: svaedir diffusion wegihts
* `--cfgvae` : vae config file
* `--gen_vae_path` : savedir vae weights


## 2. Molecular optimization with RL
### 2.1. Training with target

```
CUDA_VISIBLE_DEVICES=1 python scripts/train_rl.py path/params.yml --steps 250   --adaptive_weight --scaffold_weight 0.0  --dap_factor 1.5  --sampling   --weights dock=0.35,logp=0.25,qed=0.35,sa=0.25   --dock --receptor_path path/target.pdbqt   --box_path path/target.box.txt  --batch 128  --admet --admet_json path/admet_par.json   --flag save_dir_suffix  --admet_thr 0.9 
```
* `--dock`: enable affinity optimization
* `--receptor_path`: target pdbqt file
* `--box_path`: target enclosed box 
* `--admet` : enable admet optimization
* `--admet_json` : admet config file
* `--admet_thr` : threshold filter admet: hERG/AMES/DILI
* `--weights` : initial weights of properties

### 2.1. Training with qeppi and qrci
```
CUDA_VISIBLE_DEVICES=0 python scripts/train_rl.py path/params.yml --steps 250   --adaptive_weight --scaffold_weight 0.0  --dap_factor 1.5  --sampling --weights qed=0.35,sa=0.25,qeppi=0.35,qrci=0.15,logp=0.25  --flag save_dir_suffix  --batch 256
```
### 2.3 Generation 
```
bash exec_rl_test.sh  base_path  path/RL_weights  GPU_ID  N
```
* `base_path`: base path of project
* `path/RL_weights`: path of weights by RL optimization
* `N` : Iteration number of generation.

## Pretrained Model
You can generate molecules using pretrained models.
Download result directories containing trained models [here](https://doi.org/10.5281/zenodo.19971514) and unzip downloaded files. 
Next, please replace `yourdirectory` to your directories and rewrite `batch_size` to match your gpu capacity　in `input_data/params.yml`.
\
ex. ChEMBL 
```
#!/bin/bash

export PYTHONPATH="/yourdirectory/scripts:$PYTHONPATH"
export DGLBACKEND="pytorch"

source yourenviroment

path="/yourdirectory/results/chembl_36_20251023_r1r10w1100w900_standardized_f_dyn_oral_struct_1031"
ymlFile=$path'/input_data/params.yml'

nohup python3 generation.py ${ymlFile} --N 5 --k 10000 --gpu 0 --n_jobs 24 > $path'/generate.log' &
```

## Acknowledgement
The code is based on the following repositories:
```
https://github.com/slab-it/FRATTVAE.git
```

## License

This software is released under a custom license.

Academic use of this software is free and does not require any permission.
We encourage academic users to cite our research paper (if applicable).

For commercial use, please contact us for permission.




