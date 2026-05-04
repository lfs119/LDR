from collections import defaultdict
import os
from joblib import Parallel, delayed
import torch
import datetime
from accelerate import Accelerator
from models.unet_model import *
# from ..datasets.lightning_loader import *
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from pathlib import Path
from diffusers import DDPMScheduler
from models.custom_scheduler_uncondition import DDPMDNAPipeline
from tqdm import tqdm
from models.frattvae import FRATTVAE, FRATTVAE_Enhanced
from models.property import propLinear
from models.wrapper import PropWrapper
from utils.apps import second2date, torch_fix_seed, list2pdData
from utils.mask import create_mask
from utils.data import collate_pad_fn
from utils.metrics import batched_kl_divergence, CRITERION
from utils.preprocess import SmilesToMorganFingetPrints
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from process import generate
from utils.apps import second2date
import yaml
import numpy as np
import pandas as pd
import argparse
import os
# os.environ["DGLBACKEND"] = "pytorch"
import datetime
import gc
import pickle
import time
import warnings
import moses
warnings.simplefilter('ignore')
from torch.utils.data import Subset, DataLoader, TensorDataset
from utils.preprocess import SmilesToMorganFingetPrints
from utils.chem_metrics import ADDITIONAL_METRICS_DICT, get_additional_metrics, physchem_divergence, guacamol_fcd, get_all_metrics, METRICS_DICT

def train_loop(config, vae_config):

        # load hyperparameters
    with open(vae_config) as yml:
        params = yaml.safe_load(yml)

        # path
    result_path= params['result_path']
    data_path = params['data_path']
    frag_path = params['frag_path']

    # hyperparameters for decomposition and tree-fragments
    decomp_params = params['decomp']
    n_bits = decomp_params['n_bits']
    dupl_bits = decomp_params['dupl_bits']
    radius = decomp_params['radius']
    max_depth = decomp_params['max_depth']
    max_degree = decomp_params['max_degree']
    useChiral = decomp_params['useChiral']
    ignore_dummy = decomp_params['ignore_dummy']

    # hyperparameters for model
    model_params = params['model']
    d_model = model_params['d_model']
    d_ff = model_params['d_ff']
    num_layers = model_params['nlayer']
    num_heads = model_params['nhead']
    activation = model_params['activation']
    latent_dim = model_params['latent']
    feat_dim = model_params['feat']
    props = model_params['property']
    pnames = list(props.keys())
    ploss = model_params['ploss']
    dropout = model_params['dropout']
    n_gpu = torch.cuda.device_count()
    # hyperparameters for training
    train_params = params['train']
    # batch_size = train_params['batch_size'] // n_gpu if train_params['batch_size'] > n_gpu else 1
    batch_size =config.batch_size

    ## load data
    s = time.time()
    df = pd.read_csv(data_path)

    df_frag = pd.read_csv(frag_path)
    uni_fragments = df_frag.SMILES.tolist()
    freq_label = df_frag['frequency'].tolist()
    with open(os.path.join(result_path, 'input_data', 'dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    try:
        with open(os.path.join(result_path, 'input_data', 'csr_ecfps.pkl'), 'rb') as f:
            frag_ecfps = pickle.load(f).toarray()
            frag_ecfps = torch.from_numpy(frag_ecfps).float()
        assert frag_ecfps.shape[0] == len(uni_fragments)
        assert frag_ecfps.shape[1] == (n_bits + dupl_bits)
    except Exception as e:
       
        frag_ecfps = torch.tensor(SmilesToMorganFingetPrints(uni_fragments[1:], n_bits= n_bits, dupl_bits= dupl_bits, radius= radius, 
                                                            ignore_dummy= ignore_dummy, useChiral= useChiral, n_jobs= 12)).float()
        frag_ecfps = torch.vstack([frag_ecfps.new_zeros(1, n_bits+dupl_bits), frag_ecfps])      # padding feature is zero vector
    prop_dim = sum(list(props.values())) if pnames else None
    frag_ecfps.to(config.device)
    # train valid split
    train_data = Subset(dataset, df.loc[df.test==0].index.tolist())
    valid_data = Subset(dataset, df.loc[df.test==-1].index.tolist()) 

    # make data loader
    train_loader = DataLoader(train_data, batch_size= batch_size, shuffle= True, num_workers=6,
                              pin_memory= True, collate_fn= collate_pad_fn)
  
    val_loader = DataLoader(valid_data, batch_size= batch_size, shuffle= False, num_workers=6,
                                   pin_memory= True, collate_fn= collate_pad_fn)

    config = config
    TIME = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir),
        cpu=False
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers(TIME)

    accelerator.wait_for_everyone()


    print("len train loader", len(train_loader))
    config.device = accelerator.device
    
    num_labels = frag_ecfps.shape[0]
    vae = FRATTVAE_Enhanced(num_labels, max_depth, max_degree, feat_dim, latent_dim, 
                   d_model, d_ff, num_layers, num_heads, activation, dropout, prop_dim=0, use_gnn=False)

    vae.to(config.device)

    #load vae params
    state_dict=torch.load(config.vae_path, map_location= config.device)
    vae.load_state_dict(state_dict)
    vae.eval()

    #unet model
    model = UNet2DModel(sample_size=config.sample_size, in_channels= config.in_channels, out_channels= config.out_channels, \
                      layers_per_block=config.layers_per_block, block_out_channels=config.block_out_channels, \
                        down_block_types=config.down_block_types, up_block_types=config.up_block_types)
    model.to(config.device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000,clip_sample=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_loader) * config.epoch),
    )

    # Prepare everything
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )
    ############################
    # Training
    ############################
    best_val_loss = float('inf')
    global_step = 0

    # Now you train the model
    for epoch in range(config.epoch):
        for batch_idx, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                data = batch
                frag_indices = data[0]
                features = frag_ecfps[frag_indices.flatten().cpu()].reshape(frag_indices.shape[0], frag_indices.shape[1], -1).to(config.device)
                positions = data[1].to(config.device)
                prop = data[2].to(config.device)
                if len(data) > 3:
                    edge_index = data[3].to(config.device)
                # target = torch.hstack([frag_indices.detach(), torch.zeros(frag_indices.shape[0], 1)]).flatten().long().to(config.device)

                # make mask
                frag_indices = torch.hstack([torch.full((frag_indices.shape[0], 1), -1), frag_indices.detach().cpu()]).to(config.device)  # for super root
                src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(frag_indices, frag_indices, pad_idx= 0, batch_first= True)

                # forward
                with torch.no_grad():
                    # z, mu, ln_var, output = vae(features, positions, 
                    #                                     src_mask, src_pad_mask, 
                    #                                     tgt_mask, tgt_pad_mask)
                    z, mu, ln_var = vae.encode(features, positions, 
                                                        src_mask, src_pad_mask) 
                
                bs = z.shape[0]
                clean_images = z.reshape(bs, 1, config.sample_size, config.sample_size)         # output: shape= (B, L+1, num_labels)
                noise = torch.randn((clean_images.shape)).to(config.device)
                # Sample a random timestep for each image
                timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bs,)
                    ).long()
                timesteps = timesteps.to(config.device)

                    # Add noise to the clean images according to the noise magnitude at each timestep
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                    # Predict the noise residual
                noise_pred = model(noisy_images, timesteps,return_dict=False)[0]          

                # backward
                loss = F.mse_loss(noise_pred, noise)
              

                accelerator.backward(loss, retain_graph=True)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            logs = {"loss": loss.detach().item(),"lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            accelerator.log(logs, step=global_step)
            global_step += 1
        if accelerator.is_local_main_process:
            model.eval()
            with torch.no_grad():
                loss_li = []
                for batch_idx, val_data in enumerate(val_loader):
                    data = val_data
                    frag_indices = data[0]
                    features = frag_ecfps[frag_indices.flatten().cpu()].reshape(frag_indices.shape[0], frag_indices.shape[1], -1).to(config.device)
                    positions = data[1].to(config.device)
                    prop = data[2]
                    if len(data) > 3:
                        edge_index = data[3]
                    # target = torch.hstack([frag_indices.detach(), torch.zeros(frag_indices.shape[0], 1)]).flatten().long().to(config.device)

                    # make mask
                    frag_indices = torch.hstack([torch.full((frag_indices.shape[0], 1), -1), frag_indices.detach().cpu()]).to(config.device)  # for super root
                    src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(frag_indices, frag_indices, pad_idx= 0, batch_first= True)

                    # forward
                    with torch.no_grad():
                        # z, mu, ln_var, output = vae(features, positions, 
                        #                                     src_mask, src_pad_mask, 
                        #                                     tgt_mask, tgt_pad_mask) 
                        z, mu, ln_var = vae.encode(features, positions, 
                                                        src_mask, src_pad_mask)
                    
                    bs = z.shape[0]
                    clean_images = z.reshape(bs, 1, config.sample_size, config.sample_size)         # output: shape= (B, L+1, num_labels)
                    noise = torch.randn((clean_images.shape)).to(config.device)
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps, (bs,)
                        ).long()
                    timesteps = timesteps.to(config.device)

                        # Add noise to the clean images according to the noise magnitude at each timestep
                    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                    # Predict the noise residual
                    noise_pred = model(noisy_images, timesteps, return_dict=False)[0] #TODO need to verify
                    v_loss = F.mse_loss(noise_pred, noise)
                    loss_li.append(v_loss.mean().item())


                val_loss = sum(loss_li)/len(loss_li)
                accelerator.log({"valid_loss": val_loss}, step=global_step)

                print(f"Epoch: {epoch} / {config.epoch}, lr: {lr_scheduler.get_last_lr()[0]:.4e}, step: {global_step}, Train Loss: {loss.mean().item():.4f}, Val Loss: {val_loss:.4f}, Time: {second2date(time.time())}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.module.state_dict(), os.path.join(config.save_path, TIME+"_best_unet_model.pth"))
           
        accelerator.wait_for_everyone()
    torch.save(model.module.state_dict(), os.path.join(config.save_path, TIME+"_last_unet_model.pth"))
    print(">>> Training finished.")
    accelerator.wait_for_everyone()
    accelerator.end_training()

def evaluate(config, vae_config, vae_ckpt,unet_ckpt,N=3, k=10000):

    with open(vae_config) as yml:
        params = yaml.safe_load(yml)

        # path
    result_path= params['result_path']
    data_path = params['data_path']
    frag_path = params['frag_path']

    # hyperparameters for decomposition and tree-fragments
    decomp_params = params['decomp']
    n_bits = decomp_params['n_bits']
    dupl_bits = decomp_params['dupl_bits']
    radius = decomp_params['radius']
    max_nfrags = decomp_params['max_nfrags'] 
    max_depth = decomp_params['max_depth']
    max_degree = decomp_params['max_degree']
    useChiral = decomp_params['useChiral']
    ignore_dummy = decomp_params['ignore_dummy']

    # hyperparameters for model
    model_params = params['model']
    d_model = model_params['d_model']
    d_ff = model_params['d_ff']
    num_layers = model_params['nlayer']
    num_heads = model_params['nhead']
    activation = model_params['activation']
    latent_dim = model_params['latent']
    feat_dim = model_params['feat']
    props = model_params['property']
    pnames = list(props.keys())
    ploss = model_params['ploss']
    dropout = model_params['dropout']
    n_gpu = torch.cuda.device_count()
    # hyperparameters for training
    train_params = params['train']
    epochs = train_params['epoch']
    # batch_size = train_params['batch_size'] // n_gpu if train_params['batch_size'] > n_gpu else 1
    lr = train_params['lr']
    kl_w = train_params['kl_w']
    kl_anneal = train_params['kl_anneal']
    l_w = train_params['l_w']
    p_w = train_params['p_w']
    batch_size =config.batch_size

    df = pd.read_csv(data_path)
    df_frag = pd.read_csv(frag_path)
    uni_fragments = df_frag['SMILES'].tolist()
    # freq_list = df_frag['frequency'].tolist()

    df_frag = pd.read_csv(frag_path)
    uni_fragments = df_frag.SMILES.tolist()
    num_labels = len(uni_fragments)
    try:
        with open(os.path.join(result_path, 'input_data', 'csr_ecfps.pkl'), 'rb') as f:
            frag_ecfps = pickle.load(f).toarray()
            frag_ecfps = torch.from_numpy(frag_ecfps).float()
        assert frag_ecfps.shape[0] == len(uni_fragments)
        assert frag_ecfps.shape[1] == (n_bits + dupl_bits)
    except Exception as e:
        print(e, flush= True)
        frag_ecfps = torch.tensor(SmilesToMorganFingetPrints(uni_fragments[1:], n_bits= n_bits, dupl_bits= dupl_bits, radius= radius, 
                                                            ignore_dummy= ignore_dummy, useChiral= useChiral, n_jobs= 12)).float()
        frag_ecfps = torch.vstack([frag_ecfps.new_zeros(1, n_bits+dupl_bits), frag_ecfps])      # padding feature is zero vector
    ndummys = torch.tensor(df_frag['ndummys'].tolist()).long()
    prop_dim = sum(list(props.values())) if pnames else None
 

    unet_version = unet_ckpt.split("/")[-1]
    config = config
    TIME = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir),
        cpu=False
    )
    config.device = accelerator.device

    # only run on the main process
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers(TIME)
        
        ## load VAE
        vae = FRATTVAE_Enhanced(num_labels, max_depth, max_degree, feat_dim, latent_dim, 
                   d_model, d_ff, num_layers, num_heads, activation, dropout, prop_dim=0, use_gnn=False)

        vae.to(config.device)

        #load vae params
        if vae_ckpt is None:
             state_dict=torch.load(config.vae_path, map_location= config.device)
        else:
             state_dict=torch.load(vae_ckpt, map_location= config.device)
        vae.load_state_dict(state_dict)
        vae.eval()
        ## load UNet
        model = UNet2DModel(sample_size=config.sample_size, in_channels= config.in_channels, out_channels= config.out_channels, \
                      layers_per_block=config.layers_per_block, block_out_channels=config.block_out_channels, \
                        down_block_types=config.down_block_types, up_block_types=config.up_block_types)
        model.to(config.device)
        # load UNet params
        state_dict = torch.load(unet_ckpt, map_location= config.device)
        model.load_state_dict(state_dict)
        model.eval()

        noise_scheduler = DDPMScheduler(num_train_timesteps=1000,clip_sample=False)

        # define pipeline:
        pipe = DDPMDNAPipeline(unet = model, scheduler = noise_scheduler)
        pipe = pipe.to(config.device)

        # generate in batches:
        # batch_size = config.batch_size
        s = time.time()
        print(f'---{datetime.datetime.now()}: Generation start.---', flush= True)
        total_samples = k
        batch_size = 1000
        metric_storage = defaultdict(list)
        metric_means = {}
        METRICS, METRICS_TEST = {}, {}
        vae_model_name = os.path.splitext(os.path.basename(config.vae_path))[0]
        vae_model_dir_name = os.path.basename(os.path.dirname(config.vae_path))
        diff_model_dir = os.path.dirname(unet_ckpt)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        for i in range(N):
            torch.manual_seed(i)
            z_gen = []
            for idx in tqdm(range(total_samples//batch_size)):
                vae_z = pipe(output_type = np.array, return_dict= False, batch_size=batch_size,vae=vae)[0]  # (B, Latent)
                z_gen.append(vae_z)

            z_gen = torch.cat(z_gen, dim=0).reshape(total_samples, -1)
            z_gen_np = z_gen.cpu().numpy()
            np.save(os.path.join(diff_model_dir, f'z_gen_{vae_model_dir_name}_{vae_model_name}_{i}_{timestamp}.npy'), z_gen_np)

            dataloader = DataLoader(TensorDataset(z_gen), batch_size= batch_size, shuffle= False)

            z_list, dec_smiles, pred_list, cosines, euclids = generate(dataloader, uni_fragments, frag_ecfps, ndummys, 
                                                                    vae, None, max_nfrags, useChiral, False, 10, config.device) 
            
            # eval
            properties = Parallel(n_jobs= 12)(delayed(get_all_metrics)(s) for s in dec_smiles)
            prop_dict = {f'{key}': list(prop) for key, prop in zip(METRICS_DICT.keys(), zip(*properties))}

            add_properties = Parallel(n_jobs= 12)(delayed(get_additional_metrics)(s) for s in dec_smiles)
            add_prop_dict = {f'{key}': list(prop) for key, prop in zip(ADDITIONAL_METRICS_DICT.keys(), zip(*add_properties))}

            prop_dict = {**prop_dict, **add_prop_dict}

            df_gen = pd.DataFrame({**{'SMILES': dec_smiles, 'cosine': cosines, 'euclid': euclids}, **prop_dict})
            if pred_list:
                for j, pred in enumerate(zip(*pred_list)):
                    df_gen[f'pred{j}'] = pred

            # save results
            # with open(os.path.join(result_path, 'generate', f'z_gen_list_vae_diff_{TIME}.pkl'), 'wb') as f:
            #     pickle.dump(z_list, f)
            df_gen.to_csv(os.path.join(result_path, 'generate', f'vae_diff_{vae_model_dir_name}_{vae_model_name}_{i}_{timestamp}.csv'), index= False)
            
            smiles_train = df.SMILES.loc[df.test==0].tolist()
            smiles_test = df.SMILES.loc[df.test==1].tolist() if (df.test==1).sum() else smiles_train
            # moses
            metrics = moses.get_all_metrics(gen= dec_smiles, k= total_samples, device= config.device, test= smiles_train, train= smiles_train, n_jobs= 12)
            metrics_test = moses.get_all_metrics(gen= dec_smiles, k= total_samples, device= config.device, test= smiles_test, train= smiles_train, n_jobs= 12)
            for key in metrics.keys():
                if i == 0:
                    METRICS[key] = [metrics[key]]
                    METRICS_TEST[key] = [metrics_test[key]]
                else:
                    METRICS[key].append(metrics[key])
                    METRICS_TEST[key].append(metrics_test[key])

            # guacamol
            dec_smiles = df_gen['SMILES'].dropna().tolist()
            if i == 0:
                METRICS['div'] = [physchem_divergence(dec_smiles, smiles_train)]
                METRICS['fcd/g'] = [guacamol_fcd(dec_smiles, smiles_train)]
                METRICS_TEST['div'] = [physchem_divergence(dec_smiles, smiles_test)]
                METRICS_TEST['fcd/g'] = [guacamol_fcd(dec_smiles, smiles_test)]
            else:
                METRICS['div'].append(physchem_divergence(dec_smiles, smiles_train))
                METRICS['fcd/g'].append(guacamol_fcd(dec_smiles, smiles_train))
                METRICS_TEST['div'].append(physchem_divergence(dec_smiles, smiles_test))
                METRICS_TEST['fcd/g'].append(guacamol_fcd(dec_smiles, smiles_test))

            current_metrics = {k: np.array(v) for k, v in prop_dict.items()}
            metric_means[i] = current_metrics  # 存储当前迭代的均值
            for key in current_metrics:
                metric_storage[key].extend(current_metrics[key])

            print(f'[{i+1}/{N}]Average {", ".join([f"{key}: {np.nanmean(values):.4f}" for key, values in prop_dict.items()])} (elapsed time: {second2date(time.time()-s)})\n', flush= True)

            # print results
        print(f'moses metrics', flush= True)
        for key in METRICS.keys():
                print(f'- {key}:'.ljust(15) + f'<train> {np.nanmean(METRICS[key]):.4f} (std: {np.nanstd(METRICS[key]):.4f}), <test> {np.nanmean(METRICS_TEST[key]):.4f} (std: {np.nanstd(METRICS_TEST[key]):.4f})', flush= True)

            # save metrics
        df_metrics = pd.DataFrame(METRICS)
        df_metrics.to_csv(os.path.join(result_path, 'generate', f'metrics_diff_{TIME}.csv'), index= False)
        df_metrics = pd.DataFrame(METRICS_TEST)
        df_metrics.to_csv(os.path.join(result_path, 'generate', f'metrics_diff_test_{TIME}.csv'), index= False)

        print(f'---{datetime.datetime.now()}: Generation done. (elapsed time: {second2date(time.time()-s)})---\n', flush= True)
        
        print("\n=== Final Statistics for prop_dict ===")
        print(f"{'Metric':<20}{'Mean':<10}{'Std':<10}{'Unit':<10}")
        for key in sorted(metric_storage.keys()):
            data = np.array(metric_storage[key])
            mean = np.nanmean(data)
            std = np.nanstd(data)
            print(f"{key.ljust(20)} {mean:.4f} ± {std:.4f}")
           

# 