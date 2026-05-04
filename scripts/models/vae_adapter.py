import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

def load_npy_files(directory):
        data_list = []
        for filename in os.listdir(directory):
            if filename.endswith('.npy'):
                file_path = os.path.join(directory, filename)
                try:
                    data = np.load(file_path)
                    data_list.append(data)
                    print(f"Loaded {filename} with shape {data.shape}")
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        if data_list:
            combined = np.stack(data_list)
            combined = np.reshape(combined, (-1, combined.shape[-1]))
            return combined
        else:
            raise ValueError("No .npy files found in the directory")

class FRATTVAEPolicy(nn.Module):
   
    def __init__(self, vae_model, diff_guide=None, device="cuda"):
        super().__init__()
        self.vae = vae_model
        self.device = device
        self.diff_guide = diff_guide
        
        if self.diff_guide is not None:
            self.diff_guide_npy = load_npy_files(self.diff_guide)
            print('load diffusion files')
            self.num_guides = len(self.diff_guide_npy)
            self.guide_index = np.arange(0, len(self.diff_guide_npy))
            self.guide_weight  = np.ones(self.num_guides, dtype=np.float32)
            self.used_count = np.zeros(self.num_guides, dtype=np.int32)
        else:
            self.diff_guide_npy = None
            self.num_guides = 0
            self.guide_index = None
            self.guide_weight  = None
            self.used_count = None
    
    def reset_available(self):
        self.available_mask = np.ones(self.num_guides, dtype=bool)

    def sample_guides_no_repeat_epoch(self, batch_size):
        if not hasattr(self, "available_mask"):
            self.reset_available()

        available_idx = np.where(self.available_mask)[0]
        if len(available_idx) < batch_size:
            self.reset_available()
            available_idx = np.where(self.available_mask)[0]

        w = self.guide_weight[available_idx].astype(np.float32)
        w_sum = w.sum()
        if w_sum <= 0:
            w[:] = 1.0
            w_sum = w.sum()
        p = w / w_sum

        choose_idx = np.random.choice(available_idx,
                                      size=batch_size,
                                      replace=False,
                                      p=p)

        self.available_mask[choose_idx] = False
        self.used_count[choose_idx] += 1

        z_np = self.diff_guide_npy[choose_idx]
        z = torch.from_numpy(z_np).float().to(self.device)

        return z, choose_idx


    def sample_guides(self, batch_size):
        w = self.guide_weight.copy().astype(np.float64)
        w_sum = w.sum()
        if w_sum <= 0:  
            w[:] = 1.0
            w_sum = w.sum()
        p = w / w_sum

        idx = np.random.choice(self.guide_index,
                               size=batch_size,
                               replace=False,
                               p=p)

        self.used_count[idx] += 1

        z_np = self.diff_guide_npy[idx]         # shape: (batch_size, 256)
        z = torch.from_numpy(z_np).float().to(self.device)

        return z, idx
    
    def update_guide_weight(self, idx, reward, alpha=0.1):
       
        reward = np.asarray(reward, dtype=np.float32)

        r = np.maximum(reward, 0)

        old_w = self.guide_weight[idx]
        new_w = (1.0 - alpha) * old_w + alpha * r

        new_w = np.clip(new_w, 1e-3, None)

        self.guide_weight[idx] = new_w


    def sample(self, batch_size: int, temperature: float = 1.0, seed = None, sampling: bool = False):
        if seed is not None:
            torch.manual_seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        z_mean = torch.zeros(self.vae.latent_dim)
        z_var = torch.ones(self.vae.latent_dim)
        dist = MultivariateNormal(z_mean, temperature * z_var * torch.eye(self.vae.latent_dim))
        z = dist.sample((batch_size,)).to(device)
        # z = torch.randn(batch_size, self.vae.latent_dim, device=self.device)
        if self.diff_guide is not None:
            # z = torch.tensor(np.random.choice(self.diff_guide_npy, size=(batch_size,), replace=False), dtype=torch.float32).to(self.device)
            z, idx = self.sample_guides_no_repeat_epoch(batch_size)
        outputs, log_probs = self.vae.sequential_decode(
            z,
            frag_ecfps=self.vae.frag_ecfps,
            ndummys=self.vae.ndummys,
            max_nfrags=self.vae.max_nfrags,
            asSmiles=True,
            sampling=sampling
        )
        smiles = [s if isinstance(s, str) and s != "" else "" for s in outputs]
        del z
        return smiles, log_probs  # log_probs: (batch_size,)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("FRATTVAEPolicy unsupported forward")

    def sample_z(self, n):
        return torch.randn(n, self.vae.latent_dim).to(self.device)
