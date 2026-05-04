import argparse
import os

import rdkit

# set parser
parser = argparse.ArgumentParser(description= 'please enter paths')
parser.add_argument('yml', type= str, help= 'yml file')
parser.add_argument('--n_jobs', type= int, default= 10, help= 'the number of cpu for parallel, default 24')   # unimol
parser.add_argument('--free_n', action= 'store_true')
parser.add_argument('--unimol', action= 'store_true')
parser.add_argument('--biosynfoni', action= 'store_true')
parser.add_argument('--normalize', action= 'store_true', help= 'Normalize(min-max) properties[MW, QED, SA, NP, TPSA, BertzCT] using default norm-parameters. Properties not included in the list are not processed.')
args = parser.parse_args()
print(args)
import datetime
import time
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import pickle
import yaml
from scipy import sparse
from joblib import Parallel, delayed

import torch

from utils.data import ListDataset
from utils.apps import second2date
from utils.tree import get_tree_features
from utils.chem_metrics import normalize
from utils.preprocess import parallelMolsToBRICSfragments, smiles2mol, SmilesToMorganFingetPrints

import numpy as np
if args.unimol:
    from unimol_tools import UniMolRepr
    clf = UniMolRepr(data_type='molecule', 
                    remove_hs=False,
                    model_name='unimolv1', # avaliable: unimolv1, unimolv2
                    model_size='84m', # work when model_name is unimolv2. avaliable: 84m, 164m, 310m, 570m, 1.1B.
                    batch_size=4096,
                    )


def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

yml_file = args.yml
# yml_file = ''

print(f'---{datetime.datetime.now()}: Decomposition start.---', flush= True)
s = time.time()

## load hyperparameters
with open(yml_file) as yml:
    params = yaml.safe_load(yml)
print(f'load: {yml_file}', flush= True)

# path
result_path= params['result_path']
data_path = params['data_path']
frag_path = params['frag_path']

# hyperparameters for decomposition and tree-fragments
decomp_params = params['decomp']
min_size = decomp_params['min_size']
max_nfrags = decomp_params['max_nfrags']
n_bits = decomp_params['n_bits']
dupl_bits = decomp_params['dupl_bits']
radius = decomp_params['radius']
max_depth = decomp_params['max_depth']
max_degree = decomp_params['max_degree']
useChiral = decomp_params['useChiral']
ignore_double = decomp_params['ignore_double']
ignore_dummy = decomp_params['ignore_dummy']

# hyperparameters for model
model_params = params['model']
props = model_params['property']
pnames = list(props.keys())

## load data
df = pd.read_csv(data_path)
smiles = df.SMILES.tolist()
print(f'data: {data_path}', flush= True)
print(f'train: {sum(df.test==0)}, valid: {sum(df.test==-1)}, test: {sum(df.test==1)}', flush= True)
print(f'useChiral: {useChiral}, ignore_double: {ignore_double}, max_degree: {max_degree}, min_natoms: {min_size}, n_jobs: {args.n_jobs}\n', flush= True)

## decompose mols
mols = Parallel(n_jobs= args.n_jobs)(delayed(smiles2mol)(s) for s in smiles)

# load fragments as smiles
if frag_path:
    print(f'fragments load: {frag_path}\n', flush= True)
    df_frag = pd.read_csv(frag_path)
else:
    df_frag = []


#########  分子片段分解（核心步骤）
fragments_list, bondtypes_list, bondMapNums_list \
, recon_flag, uni_fragments, freq_label = parallelMolsToBRICSfragments(mols,
                                                                       minFragSize = min_size, maxFragNums= max_nfrags, maxDegree= max_degree,
                                                                       useChiral= useChiral, ignore_double= ignore_double, 
                                                                       df_frag= df_frag, asFragments= False,
                                                                       n_jobs= args.n_jobs, verbose= 0)
frag_lens = list(map(len, fragments_list))
recon_flag = np.array(recon_flag)
if len(df_frag) != len(uni_fragments):
    print(f'class increment: {len(df_frag)} -> {len(uni_fragments)}', flush= True)
    df_frag = pd.DataFrame({'SMILES': uni_fragments, 'frequency': freq_label, 'ndummys': [f.count('*') for f in uni_fragments]})
    frag_path = os.path.join(result_path, 'input_data', 'fragments.csv')
    df_frag.to_csv(frag_path, index= False)
    params['frag_path'] = frag_path

if not np.all(recon_flag>0):
    # if there are any compounds that are not reconstructable
    df = df.loc[recon_flag>0].reset_index(drop= True)
    df = df.assign(nfrags= frag_lens, recon= recon_flag[recon_flag>0])
    data_path = os.path.join(result_path, 'input_data', 'SMILES_inputed.csv')
    df.to_csv(data_path, index= False)
    params['data_path'] = data_path
    print('data_path changed:', data_path, flush= True)
else:
    df_tmp = pd.DataFrame({'idx': df.index.tolist(), 'nfrags': frag_lens, 'recon': recon_flag[recon_flag>0]})
    df_tmp.to_csv(os.path.join(result_path, 'input_data', 'num_nodes.csv'), index= False)
    del df_tmp

if sum(recon_flag>1) == 0:
    params['decomp']['useChiral'] = False     # no stereo chemistry

with open(yml_file, 'w') as yf:
    yf.write(f'# Date of update: {datetime.datetime.now()}\n')
    yaml.dump(params, yf, default_flow_style= False, sort_keys= False)

print(f'reconstruct2D: {sum(recon_flag>0)/len(smiles):.4f} ({sum(recon_flag>0)}/{len(smiles)})', flush= True)
print(f'reconstruct3D: {sum(recon_flag==3)/sum(recon_flag>1):.4f} ({sum(recon_flag==3)}/{sum(recon_flag>1)})', flush= True)


###### molformer
unimol_feat = []
if args.unimol:
    frag_ecfps = SmilesToMorganFingetPrints(uni_fragments[1:], n_bits= n_bits, dupl_bits= dupl_bits, radius= radius, 
                                                 ignore_dummy= ignore_dummy, useChiral= useChiral, n_jobs= args.n_jobs, unimol=True, biosynfoni=args.biosynfoni)
    print(f'fragment length: {len(frag_ecfps)}')
    for batch in batch_generator(frag_ecfps, batch_size=2048):
        unimol_repr = clf.get_repr(batch, return_atomic_reprs=False)
        unimol_feat.append(unimol_repr['cls_repr'])
        del unimol_repr
        torch.cuda.empty_cache() 

    frag_ecfps = np.concatenate(unimol_feat, axis= 0)
else:
    # fragments to ECFP
    frag_ecfps = np.array(SmilesToMorganFingetPrints(uni_fragments[1:], n_bits= n_bits, dupl_bits= dupl_bits, radius= radius, 
                                                 ignore_dummy= ignore_dummy, useChiral= useChiral, n_jobs= args.n_jobs, unimol=False, biosynfoni=args.biosynfoni))


if args.biosynfoni:
    n_bits = 39
frag_ecfps = np.vstack([np.zeros((1, n_bits+dupl_bits)), frag_ecfps])      # padding feature is zero vector
assert frag_ecfps.shape[0] == len(uni_fragments)
#  片段指纹编码（ECFP/Morgan）
csr_ecfps = sparse.csr_matrix(frag_ecfps) # (33504, 2048) -> (33504, 2048)
with open(os.path.join(result_path, 'input_data', 'csr_ecfps.pkl'), 'wb') as f:
    pickle.dump(csr_ecfps, f)

# make dataset
if pnames:
    if args.normalize:
        prop = np.array([normalize(df[p].to_numpy(), p) for p in pnames]).T.tolist()
    else:
        prop = df[pnames].to_numpy().tolist()
    prop_dim = sum(list(props.values()))
else:
    prop = [[float('nan')] for _ in range(len(df))]
    prop_dim = None
# 构建层次化树结构特征                  输入：fragment 列表、bond 类型、map 编号
features = Parallel(n_jobs= args.n_jobs)(delayed(get_tree_features)(f, torch.zeros(len(f), 1).float(), b, m, max_depth, max_degree, args.free_n) for f, b, m in zip(fragments_list, bondtypes_list, bondMapNums_list))
frag_indices, _, positions, edge_index = zip(*features)

# frag_indices: 每个 fragment 在 uni_fragments 中的索引
# positions: 在树结构中的位置编码（如层级、兄弟顺序）

prop = torch.tensor(prop).reshape(len(df), -1).float()

dataset = ListDataset(frag_indices, positions, prop)
# dataset = ListDatasetEdgeIndex(frag_indices, positions, prop, edge_index)

assert len(df) == len(dataset)
with open(os.path.join(result_path, 'input_data', 'dataset.pkl'), 'wb') as f:
    pickle.dump(dataset, f)

print(f'rdkit version: {rdkit.__version__}')
print(f'fragments: {len(uni_fragments)}, feature: {n_bits+dupl_bits}, degree: {df_frag.ndummys.max()}, props: {pnames}', flush= True)
print(f'fragments per mol: {min(frag_lens)} - {max(frag_lens)} (mean: {np.mean(frag_lens):.1f}, median: {np.median(frag_lens):.1f}), mol as a fragment: {frag_lens.count(1)}', flush= True)
print(f'---{datetime.datetime.now()}: Decomposition done. (elapsed time: {second2date(time.time()-s)})---\n', flush= True)
