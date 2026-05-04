import argparse, os, sys
import time, datetime
import yaml, easydict
import torch
# Datasets
# Sessions
import diffusion_unet as diff_unet
import torch.multiprocessing as mp
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class LoaderConfig():
    def __init__(self, loader_cfg=None):
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), loader_cfg.data_path)

        if loader_cfg is None:
            self.num_workers: int = 2 # cpus
            self.num_processes: int = 8 # gpus
            self.batch_size: int = 12
            self.train_prop: float = 0.5
            self.valid_prop: float = 0.1
            self.shuffle: bool = True
            self.generator = torch.Generator().manual_seed(50)
        else:
            self.num_workers: int = loader_cfg.num_workers
            self.num_processes: int = loader_cfg.num_processes
            self.batch_size: int = loader_cfg.batch_size
            self.train_prop: float = loader_cfg.train_prop
            self.valid_prop: float = loader_cfg.valid_prop
            self.shuffle: bool = loader_cfg.shuffle
            self.generator = torch.Generator().manual_seed(loader_cfg.seed)

class TrainerConfig():
    def __init__(self, save_path, trainer_cfg=None, loader_cfg=None, model_cfg=None):
        # self.loader_config = LoaderConfig(loader_cfg=loader_cfg)
        self.save_path = save_path

        # training parameters
        for attr in trainer_cfg:
            self.__setattr__(attr, trainer_cfg[attr])
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        # model parameters
        for attr in model_cfg:
            self.__setattr__(attr, model_cfg[attr])

def build_parser(root):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='diff_unet', help='model to train')
    parser.add_argument('--dataset', type=str, default='reference', help='dataset to train on')
    parser.add_argument('--cfgdir', type=str, default=root+'/src/configs', help='path to config file')
    parser.add_argument('--cfgvae', type=str, default=root+'/src/configs', help='path to config file')
    parser.add_argument('--cfgname', type=str, default='', help='name of config file')
    parser.add_argument('--cptname', type=str, default='', help='name of checkpoint file')
    parser.add_argument('--N', type=int, default=3, help='number of iteration')
    parser.add_argument('--tgtdir', type=str, default=root+'/saved_models', help='path to save model')
    parser.add_argument('--gen_vae_path', type=str, default=None, help='path to load vae model')
    parser.add_argument('--gen_unet_path', type=str, default=root+'/saved_models', help='path to load unet model')
    return parser

def prepare_config(config_path, save_path):
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = easydict.EasyDict(cfg)
    print(cfg)
    config = TrainerConfig(save_path, cfg['trainer_cfg'], model_cfg=cfg['model_cfg'])

    return config

if __name__ == "__main__":
    parser = build_parser(os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    # path = os.path.join(args.cfgdir, args.model + '.yaml') if not args.cfgname else os.path.join(args.cfgdir, args.cfgname + '.yaml')
    path = os.path.join(args.cfgdir) if not args.cfgname else os.path.join(args.cfgdir, args.cfgname + '.yaml')

    config = prepare_config(path, args.tgtdir)
    # create the target directory if it does not exist
    if not os.path.exists(args.tgtdir):
        os.makedirs(args.tgtdir)

    if args.model == 'diff_unet':
        diff_unet.train_loop(config, args.cfgvae)
    if args.model == 'generate':
        # args.cptname: the checkpoint path of UNet
        mp.set_start_method("spawn")
        diff_unet.evaluate(config,args.cfgvae,args.gen_vae_path,args.gen_unet_path, N=args.N)
