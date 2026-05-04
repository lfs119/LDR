import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from moses.metrics.SA_Score import sascorer
from moses.metrics.NP_Score import npscorer
from moses.metrics.utils import qeppi
from moses.metrics.utils import qrci

from tqdm import tqdm

from dataclasses import dataclass

@dataclass
class RLConfig:
    batch_size: int = 64
    steps: int = 20000
    temperature: float = 1.0
    entropy_beta: float = 0.01
    lr: float = 1e-5
    max_len: int = 140
    device: str = "cuda"
    save_every: int = 1000
    n_jobs: int = 6

def compute_reward(smiles: str, qed_w=0.4, sa_w=0.4, logp_w=0.2, qeppi_w=0.4, np_w=0.3, qrci_w=0.3) -> float:
    if not smiles:
        return 0.0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0

    # QED: [0,1], higher better
    qed = QED.qed(mol)

    # lower better → normalize to [0,1]
    sa_norm = sascorer.calculateScore(mol) / 10.0

    # LogP: target ~2.5, Gaussian reward
    logp = Descriptors.MolLogP(mol)
    logp_score = np.exp(-0.5 * ((logp - 2.5) / 1.0) ** 2)

    # QEPPIs: [0,1], higher better
    qeppi_s = qeppi(smiles) 

    np_s = npscorer.scoreMol(mol)

    qrci_s = qrci(smiles)

    reward = qed_w * qed + sa_w * sa_norm + logp_w * logp_score + qeppi_w * qeppi_s + np_w * np_s + qrci_w * qrci_s
    # return min(max(reward, 0.0), 1.0)
    return reward


class LatentPolicy(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, z0):
        delta = self.net(z0)
        return z0 + delta, delta  # return z and offset
    
class FRATTVAERLOptimizer:
    def __init__(
        self,
        vae_model,
        frag_ecfps,
        ndummys,
        labels,
        device="cuda",
        lr_policy=1e-4,
        lr_vae=1e-5,
        max_nfrags=30,
        free_n=False,
        qed_w=0.4,
        sa_w=0.4,
        logp_w=0.2,
        grad_clip=1.0
    ):
        self.vae = vae_model.to(device).eval()
        self.frag_ecfps = frag_ecfps.to(device)
        self.ndummys = ndummys.to(device)
        self.vae.set_labels(labels)
        self.device = device
        self.max_nfrags = max_nfrags
        self.free_n = free_n
        self.grad_clip = grad_clip

        # Policy
        self.policy = LatentPolicy(latent_dim=vae_model.latent_dim).to(device)
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)

        # VAE optimizer (for Stage 2)
        self.opt_vae = torch.optim.Adam(
            list(self.vae.decoder.parameters()) +
            list(self.vae.fc_dec.parameters()) +
            list(self.vae.fc_memory.parameters()),
            lr=lr_vae
        )

        # Reward weights
        self.reward_weights = (qed_w, sa_w, logp_w)

        # Moving average baseline
        self.baseline = 0.0
        self.baseline_decay = 0.95

        self.stage = 1  # 1 or 2

    def set_stage(self, stage: int):
        assert stage in [1, 2]
        self.stage = stage
        # Freeze encoder in both stages
        for p in self.vae.encoder.parameters():
            p.requires_grad = False
        for p in self.vae.fc_vae.parameters():
            p.requires_grad = False

        if stage == 1:
            # Freeze decoder
            for p in self.vae.decoder.parameters():
                p.requires_grad = False
            for p in self.vae.fc_dec.parameters():
                p.requires_grad = False
            for p in self.vae.fc_memory.parameters():
                p.requires_grad = False
        else:
            # Unfreeze decoder
            for p in self.vae.decoder.parameters():
                p.requires_grad = True
            for p in self.vae.fc_dec.parameters():
                p.requires_grad = True
            for p in self.vae.fc_memory.parameters():
                p.requires_grad = True

    def generate_smiles(self, batch_size=32):
        z0 = torch.randn(batch_size, self.vae.latent_dim).to(self.device)
        z, delta = self.policy(z0)

        with torch.set_grad_enabled(self.stage == 2):
            smiles_list = self.vae.sequential_decode(
                z,
                frag_ecfps=self.frag_ecfps,
                ndummys=self.ndummys,
                max_nfrags=self.max_nfrags,
                free_n=self.free_n,
                asSmiles=True,
                conditions=None
            )
        return smiles_list, delta

    def train_step(self, batch_size=32):
        smiles_list, delta = self.generate_smiles(batch_size)

        # Compute rewards
        rewards = []
        valid_mask = []
        for s in smiles_list:
            r = compute_reward(s, *self.reward_weights)
            rewards.append(r)
            valid_mask.append(float(r > 0))

        rewards = torch.tensor(rewards, device=self.device)
        valid_mask = torch.tensor(valid_mask, device=self.device)

        # Update baseline
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * rewards.mean().item()

        # REINFORCE loss: only on valid molecules
        adv = rewards - self.baseline
        loss_rl = -(adv * (delta.norm(dim=1) ** 2) * valid_mask).mean()

        # Optimize
        self.opt_policy.zero_grad()
        if self.stage == 2:
            self.opt_vae.zero_grad()

        loss_rl.backward()

        # Clip gradients
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        if self.stage == 2:
            nn.utils.clip_grad_norm_(self.vae.decoder.parameters(), self.grad_clip)
            nn.utils.clip_grad_norm_(self.vae.fc_dec.parameters(), self.grad_clip)
            nn.utils.clip_grad_norm_(self.vae.fc_memory.parameters(), self.grad_clip)

        self.opt_policy.step()
        if self.stage == 2:
            self.opt_vae.step()

        # Logging
        validity = valid_mask.mean().item()
        avg_reward = rewards.mean().item()
        return loss_rl.item(), avg_reward, validity, self.baseline
    













# # Load your pretrained FRATTVAE
# vae = FRATTVAE(...)
# vae.load_state_dict(torch.load("frattvae_pretrained.pt"))

# # Load fragment vocab
# frag_ecfps = torch.load("frag_ecfps.pt")      # (N, 2048)
# ndummys = torch.load("ndummys.pt")            # (N,)
# labels = np.load("labels.npy").tolist()       # [str, ...]

# # Create optimizer
# rl_opt = FRATTVAERLOptimizer(
#     vae, frag_ecfps, ndummys, labels,
#     device="cuda",
#     lr_policy=1e-4,
#     lr_vae=5e-6
# )

# # Stage 1: Fixed VAE
# rl_opt.set_stage(1)
# for epoch in tqdm(range(500)):
#     loss, reward, validity, baseline = rl_opt.train_step(batch_size=64)
#     if epoch % 50 == 0:
#         print(f"Stage1 | Epoch {epoch} | Reward: {reward:.4f} | Valid: {validity:.2%}")

# # Stage 2: Finetune Decoder
# rl_opt.set_stage(2)
# for epoch in tqdm(range(300)):
#     loss, reward, validity, baseline = rl_opt.train_step(batch_size=64)
#     if epoch % 50 == 0:
#         print(f"Stage2 | Epoch {epoch} | Reward: {reward:.4f} | Valid: {validity:.2%}")

# # Save policy
# torch.save(rl_opt.policy.state_dict(), "frattvae_policy.pt")