import datetime
import pickle
import time
import pandas as pd
import torch, json
import argparse
import os

import yaml
from models.vae_adapter import FRATTVAEPolicy
from models.rl_main import PROTrainer
from models.rl_utils import RLConfig
from models.frattvae import FRATTVAE_Enhanced 
import warnings

from utils.apps import second2date
warnings.simplefilter('ignore')
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

# print(">>> LOADING CORRECT train_rl.py <<<")
# print("__file__ =", __file__)

def parse_weight_dict(weight_str):
    """wight dict from string"""
    weight_dict = {}
    for pair in weight_str.split(','):
        key, value = pair.split('=')
        weight_dict[key] = float(value)
    return weight_dict


parser = argparse.ArgumentParser(
    description="Reinforcement learning for molecular generation with docking and ADMET scoring."
)

# Positional argument: YAML config file
parser.add_argument(
    'yml',
    type=str,
    help="Path to the YAML configuration file (e.g., params.yml)"
)

# Docking-related
parser.add_argument(
    "--receptor_path",
    type=str,
    default='data/hDAT.pdbqt',
    help="Path to receptor file in PDBQT format (default: data/hDAT.pdbqt)"
)
parser.add_argument(
    "--box_path",
    type=str,
    default='data/hDAT.box.txt',
    help="Path to docking box configuration file (default: data/hDAT.box.txt)"
)

parser.add_argument(
    "--receptor_path_dual",
    type=str,
    default= None,
    help="Path to receptor file in PDBQT format (default: data/hDAT.pdbqt)"
)
parser.add_argument(
    "--box_path_dual",
    type=str,
    default= None,
    help="Path to docking box configuration file (default: data/hDAT.box.txt)"
)

parser.add_argument(
    "--GPU_Docking",
    action="store_true",
    help="Enable GPU-accelerated docking"
)

parser.add_argument('--target_pair', type=str, default='gsk3b_jnk3', help='Target pair for docking')


# ADMET
parser.add_argument(
    "--admet_json",
    type=str,
    default=None,
    help="Path to JSON file containing ADMET property weights (e.g., admet_par.json)"
)
parser.add_argument(
    "--admet",
    action="store_true",
    help="Enable ADMET multi-property scoring during RL"
)

parser.add_argument(
    "--admet_thr",
    type=float, default=1.0,
    help="Enable ADMET multi-property filter scoring  during RL"
)

# Reinforcement Learning
parser.add_argument(
    "--adaptive_weight",
    action="store_true",
    help="Enable dynamic (adaptive) weighting of reward components"
)
parser.add_argument(
    '--batch', type=int, default=128,
    help='Batch size for policy gradient updates (default: 128)'
)
parser.add_argument(
    '--steps', type=int, default=350,
    help='Number of training steps (default: 350)'
) # 
parser.add_argument(
    '--save_every_n_epochs', type=int, default=50,
    help='Number of training save epoche gap (default: 50)'
) # save_every_n_epochs
parser.add_argument(
    '--weights', type=str,
    default="qed=0.35,sa=0.25,qeppi=0.01,np=0.45,qrci=0.15",
    help='Comma-separated reward weights in format: key1=val1,key2=val2,... '
         '(e.g., "dock=0.3,qed=0.4,sa=0.3"). These override values from --admet_json.'
)

# Loss weights
parser.add_argument('--scaffold_weight', type=float, default=0.5, help='Weight for scaffold diversity loss')
parser.add_argument('--diversity_lambda', type=float, default=0.1, help='Weight for molecular diversity loss')
parser.add_argument('--kl_lambda', type=float, default=0.02, help='Weight for KL divergence loss')
parser.add_argument('--dap_factor', type=float, default=2.0, help='Dynamic Adaptation Policy scaling factor')

# Misc
parser.add_argument('--n_jobs', type=int, default=16, help='Number of CPU cores for parallel computation')
parser.add_argument('--flag', type=str, default="", help='Custom experiment tag or identifier')
parser.add_argument("--BBB", action="store_true", help="Include blood-brain barrier penetration in scoring")
parser.add_argument("--sampling", action="store_true", help="Run in sampling mode (no training)")
parser.add_argument("--dock", action="store_true", help="Enable molecular docking reward")
parser.add_argument('--diff_guide', type=str, default=None,  help='training RL with diffusion guide')
parser.add_argument('--safe_item', action="store_true", help="Enable min weight for hERG,AMES,DILI")
parser.add_argument('--var_design', action="store_true", help="inverse variance weighting or variance weighting")




args = parser.parse_args()
smi_weights = parse_weight_dict(args.weights)
if args.admet_json is not None:
    admet_par = json.load(open(args.admet_json))
    if isinstance(admet_par, dict):
            smi_weights.update(admet_par)
    else:
            raise ValueError(f"JSON file must contain a dictionary, got {type(admet_par)}")
else:
    admet_json = None
# print(f'args: {args}')

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    yml_file = args.yml
    ## load hyperparameters
    with open(yml_file) as yml:
        params = yaml.safe_load(yml)
    print(f'load: {yml_file}', flush= True)

    frag_path = params['frag_path']
    result_path= params['result_path']
    decomp_params = params['decomp']
    n_bits = decomp_params['n_bits']
    max_nfrags = decomp_params['max_nfrags'] 
    dupl_bits = decomp_params['dupl_bits']
    max_depth = decomp_params['max_depth']
    max_degree = decomp_params['max_degree']
   

    # hyperparameters for model
    model_params = params['model']
    d_model = model_params['d_model']
    d_ff = model_params['d_ff']
    num_layers = model_params['nlayer']
    num_heads = model_params['nhead']
    activation = model_params['activation']
    latent_dim = model_params['latent']
    feat_dim = model_params['feat']

    df_frag = pd.read_csv(frag_path)
    uni_fragments = df_frag['SMILES'].tolist()
    try:
        with open(os.path.join(result_path, 'input_data', 'csr_ecfps.pkl'), 'rb') as f:
            frag_ecfps = pickle.load(f).toarray()
            frag_ecfps = torch.from_numpy(frag_ecfps).float()
        assert frag_ecfps.shape[0] == len(uni_fragments)
        assert frag_ecfps.shape[1] == (n_bits + dupl_bits)
    except Exception as e:
        print(e, flush= True)
       
    ndummys = torch.tensor(df_frag['ndummys'].tolist()).long()
    num_labels = frag_ecfps.shape[0]

    start = time.time()
    print(f'---{datetime.datetime.now()}: start.---', flush= True)


    vae = FRATTVAE_Enhanced(num_labels, max_depth, max_degree, feat_dim, latent_dim, 
                   d_model, d_ff, num_layers, num_heads, activation, prop_dim=0, use_gnn=False)
    vae.load_state_dict(torch.load(os.path.join(result_path, 'models', f'model_best.pth'), map_location= device))
    vae.PE._update_weights() 
    

   
    for param in vae.parameters():
        param.requires_grad = False  

    # 解冻 decoder 
    for param in vae.decoder.parameters():
        param.requires_grad = True
    for param in vae.fc_dec.parameters():
        param.requires_grad = True
    for param in vae.fc_memory.parameters():
        param.requires_grad = True

    # for param in vae.embed.parameters(): param.requires_grad = True

    vae.to(device)
    vae.train()

    # 检查是否有可训练参数
    trainable_params = [p for p in vae.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")

    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters! Did you forget to unfreeze the decoder?")
    

    # 加载fragment数据
    vae.set_labels(uni_fragments)
    vae.frag_ecfps = frag_ecfps
    vae.ndummys = ndummys
    vae.max_nfrags = max_nfrags

    policy = FRATTVAEPolicy(vae, diff_guide=args.diff_guide, device=device).to(device)

    # weights = {"qed":0.35,"sa":0.25,"qeppi":0.01,"np":0.45,"qrci":0.15}
    cfg = RLConfig(batch_size=args.batch, steps=args.steps, temperature=1.0, lr=1e-5, save_every=args.save_every_n_epochs, n_jobs=args.n_jobs)
    cfg.sampling = args.sampling
    cfg.target_pair = args.target_pair
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.BBB:
        admet_project = 'oral_cns'
    else:
        admet_project = 'oral_pg'
    trainer = PROTrainer(policy, cfg, smi_weights, save_dir=f"runs/{admet_project}_{args.flag}", 
                         diversity_lambda=args.diversity_lambda, scaffold_weight=args.scaffold_weight, kl_lambda=args.kl_lambda, 
                         dap_factor=args.dap_factor, w_flag=args.adaptive_weight, dock=args.dock, admet=args.admet, admet_project=admet_project,
                        receptor_path=args.receptor_path, box_path=args.box_path, receptor_path_dual=args.receptor_path_dual, box_path_dual=args.box_path_dual, admet_thr=args.admet_thr, safe_item=args.safe_item,variance_design=args.var_design, GPU_Docking=args.GPU_Docking)

    for i in range(cfg.steps):
        stats = trainer.step(i)
        torch.cuda.empty_cache()
        if (i+1)%50==0:
            print(stats)

    print(f'---{datetime.datetime.now()}: all process done. (elapsed time: {second2date(time.time()-start)})---\n', flush= True)

if __name__ == "__main__":
    main()
