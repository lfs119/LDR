import datetime
import os, json, csv, torch
from torch.optim import AdamW
from typing import Dict, List

from .parallel_docking_multi import MultiTargetDockingPool
from .parallel_docking_multi_gpu import MultiTargetDockingPoolGPU

from .parallel_docking import DockingPool, load_box

from .scaffold import ScaffoldMemory
# from .scoring import multiobjective_vector, multiobjective_vector_jnk3
from .scoring import multiobjective_vector, multiobjective_vector_dat, multiobjective_vector_dat_admet, multiobjective_vector_dat_dual

from .diversity import DiversityMemory
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

class MovingAverage:
    def __init__(self, momentum=0.95):
        self.m = momentum
        self.v = None
    def update(self, x):
        self.v = x if self.v is None else self.m*self.v + (1-self.m)*x
        return self.v

class TopK:
    def __init__(self, k=50):
        self.k = k
        self.data = []
    def add(self, item):
        self.data.append(item)
        self.data.sort(key=lambda t: t[0], reverse=True)
        if len(self.data) > self.k: self.data.pop()

# ---------- Pareto + crowding ----------
def pareto_sort(vectors: List[List[float]]):
    N = len(vectors)
    dominates = [[] for _ in range(N)]
    dominated_count = [0]*N
    fronts = [[]]
    for i in range(N):
        for j in range(N):
            if i==j: continue
            if all(vectors[i][k] >= vectors[j][k] for k in range(len(vectors[i]))) and \
               any(vectors[i][k] > vectors[j][k] for k in range(len(vectors[i]))):
                dominates[i].append(j)
            elif all(vectors[j][k] >= vectors[i][k] for k in range(len(vectors[i]))) and \
                 any(vectors[j][k] > vectors[i][k] for k in range(len(vectors[i]))):
                dominated_count[i] += 1
        if dominated_count[i]==0:
            fronts[0].append(i)
    rank = [0]*N
    f=0
    while fronts[f]:
        next_front=[]
        for i in fronts[f]:
            for j in dominates[i]:
                dominated_count[j]-=1
                if dominated_count[j]==0:
                    rank[j]=f+1
                    next_front.append(j)
        f+=1
        fronts.append(next_front)
    return rank, fronts[:-1]


def crowding_distance(front_vectors: List[List[float]]):
    n = len(front_vectors)
    if n == 0: return np.array([])
    m = len(front_vectors[0])
    distance = np.zeros(n)
    arr = np.array(front_vectors)
    for k in range(m):
        order = np.argsort(arr[:,k])
        # distance[order[0]] = distance[order[-1]] = np.inf
        norm = arr[:,k].max() - arr[:,k].min() + 1e-8
        max_dist = (arr[order[-1], k] - arr[order[0], k]) / norm if n > 2 else 1.0
        distance[order[0]] += max_dist
        distance[order[-1]] += max_dist
        for i in range(1, n-1):
            distance[order[i]] += (arr[order[i+1],k] - arr[order[i-1],k]) / norm
    
    distance = np.nan_to_num(distance, nan=0.0, posinf=1e6, neginf=0.0)
    return distance


# ---------- Adaptive Weighting ----------
class AdaptiveWeighting:
    def __init__(self, init_weights: Dict[str,float], variance_design=False, alpha=0.3):
        self.weights = dict(init_weights)
        self.alpha = alpha
        self.variance_design = variance_design

    def update(self, batch_props: List[Dict[str,float]]):
        keys = list(self.weights.keys())
        arr = {k: [] for k in keys}
        for p in batch_props:
            for k in keys:
                if k in p:
                    arr[k].append(p[k])
        vars_ = {k: np.var(arr[k]) + 1e-6 for k in keys}
        if not self.variance_design:
             inv = {k: 1.0/v for k,v in vars_.items()}
        else:
            inv = {k: v for k,v in vars_.items()}
        total = sum(inv.values())
        new_w = {k: inv[k]/total for k in keys}
        for k in keys:
            self.weights[k] = (1-self.alpha)*self.weights[k] + self.alpha*new_w[k]
        return self.weights


# ---------- Visualization Logger ----------
class FrontierLogger:
    def __init__(self, save_dir, keys):
        self.save_dir = os.path.join(save_dir, "plots")
        os.makedirs(self.save_dir, exist_ok=True)
        self.keys = keys
        self.history = []

    def log_frontier(self, step, vectors, ranks, smiles, props_list):
        arr = np.array(vectors)
        ranks = np.array(ranks)
        data = {"step": step, "vectors": arr.tolist(), "ranks": ranks.tolist()}
        self.history.append(data)
        with open(os.path.join(self.save_dir, "frontier_log.json"), "w") as f:
            json.dump(self.history, f, indent=2)
        self._plot_frontier(step, arr, ranks)
        # self._save_top3(step, arr, ranks, smiles, props_list)
        self._save_top10(step, arr, ranks, smiles, props_list)

    def _plot_frontier(self, step, arr, ranks):
        if arr.shape[1] < 2: return
        plt.figure(figsize=(6,5))
        sc = plt.scatter(arr[:,0], arr[:,1], c=ranks, cmap="viridis", alpha=0.7)
        plt.colorbar(sc, label="Pareto Rank")
        plt.xlabel(self.keys[0]); plt.ylabel(self.keys[1])
        plt.title(f"Pareto Frontier (step={step})")
        plt.tight_layout()
        path = os.path.join(self.save_dir, f"frontier_step_{step}.png")
        plt.savefig(path)
        plt.close()
        return path

    def _save_top3(self, step, arr, ranks, smiles, props_list):
        rank0_idx = np.where(np.array(ranks)==0)[0]
        if len(rank0_idx)==0: return
        top_idx = rank0_idx[:3]
        tops = []
        for i in top_idx:
            tops.append({
                "smiles": smiles[i],
                "rank": int(ranks[i]),
                "props": props_list[i]
            })
        with open(os.path.join(self.save_dir, f"frontier_top3_step_{step}.json"), "w") as f:
            json.dump(tops, f, indent=2)

    def _save_top10(self, step, arr, ranks, smiles, props_list):
        rank0_idx = np.where(np.array(ranks)==0)[0]
        if len(rank0_idx)==0: return
        top_idx = rank0_idx[:10]
        tops = []
        for i in top_idx:
            tops.append({
                "smiles": smiles[i],
                "rank": int(ranks[i]),
                "props": props_list[i]
            })
        with open(os.path.join(self.save_dir, f"frontier_top10_step_{step}.json"), "w") as f:
            json.dump(tops, f, indent=2)


# ---------- CSV Logger ----------
class CSVLogger:
    def __init__(self, save_dir, fieldnames):
        today = datetime.datetime.today()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(save_dir, f"train_log_{timestamp}.csv")
        # self.path = os.path.join(save_dir, "train_log.csv")
        self.fieldnames = fieldnames
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: dict):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


# ---------- PRO Trainer ----------       kl_lambda=0.02
class PROTrainer:
    def __init__(self, policy, cfg, weights: dict, property_targets=None,
                 save_dir="runs/exp_pro", diversity_lambda=0.1, kl_lambda=0.02, crowd_alpha=0.2, ppo_eps=0.2, scaffold_weight=0.3,
                 dap_patience=10,        
                 dap_factor=1.5,         # 调整 kl_lambda 的倍数
                 kl_lambda_min=0.01,     # kl_lambda 下限
                 kl_lambda_max=0.5,
                 w_flag=False,
                 dock=False,
                 admet=False,
                 admet_project="oral_peripheral",
                 receptor_path=None,
                 box_path = None,
                 receptor_path_dual = None,
                 box_path_dual = None,
                 admet_thr = 1.0,
                 safe_item = False,
                 variance_design = False,
                 GPU_Docking = False):
        self.policy = policy
        self.cfg = cfg
        self.weights = dict(weights)
        self.targets = property_targets or {}
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.optim = AdamW(self.policy.parameters(), lr=cfg.lr)
        self.baseline = MovingAverage(momentum=0.98)
        self.mem = DiversityMemory()        # 指纹
        self.scaffold_mem = ScaffoldMemory()  # 骨架
        self.scaffold_weight = scaffold_weight
        self.div_lambda = diversity_lambda
        # self.kl_lambda = kl_lambda
        self.crowd_alpha = crowd_alpha
        self.topk = TopK(k=100)
        self.old_policy = None
        self.adapt = AdaptiveWeighting(weights, variance_design)
        self.logger = FrontierLogger(save_dir, list(weights.keys()))
        self.writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard"))
        # add 
        fields = ["step","loss","reward_mean","reward_max","KL","baseline","uniqueness","kl_lambda","dap_factor","scaffold_weight"] + [f"w_{k}" for k in weights.keys()]
        self.csv = CSVLogger(save_dir, fields)
        self.ppo_eps = ppo_eps
        self.smiles_freq = defaultdict(int)
            # >>> DAP 状态 <<<
        self.dap_patience = dap_patience
        self.dap_factor = dap_factor
        self.kl_lambda_min = kl_lambda_min
        self.kl_lambda_max = kl_lambda_max
        self.kl_lambda = max(kl_lambda_min, min(kl_lambda_max, kl_lambda))  

        self.dap_reward_history = []      #  reward_mean
        self.dap_uniq_history = []        # uniqueness
        self.dap_last_explore = -1        
        self.w_flag = w_flag
        self.dock = dock
        self.admet = admet
        self.GPU_Docking = GPU_Docking
        self.target_pair = cfg.target_pair
        if self.admet:
            from admet_ai import ADMETModel
            self.admet_model = ADMETModel()
            self.admet_thr = admet_thr

        self.admet_project = admet_project

        self.receptor_path_dual = receptor_path_dual
        self.box_path_dual = box_path_dual

        if self.dock:
            try:
                self.receptor = receptor_path
                self.center, self.size = load_box(box_path)
            except Exception as e:
                raise ValueError(f"Failed to load box file: {e}")
        else:
            self.receptor = None
            self.center = None
            self.size = None
        
        if self.receptor_path_dual is not None and self.box_path_dual is not None:
            try:
                self.receptor_dual = receptor_path_dual
                self.center_dual, self.size_dual = load_box(box_path_dual)
            except Exception as e:
                raise ValueError(f"Failed to load dual box file: {e}")

        self.mean_reward = 0.0
        self.safe_item = safe_item
        self.variance_design = variance_design
    
    @torch.no_grad()
    def backup_old_policy(self):
        # self.old_policy = {k: v.clone() for k, v in self.policy.state_dict().items()}   # detach
        self.old_policy_params = [p.clone().detach() for p in self.policy.parameters()]
    
    @torch.no_grad()
    def backup_old_policy_copy(self):
        self.old_policy = np.copy.deepcopy(self.policy)
        self.old_policy.eval()  

    def kl_divergence(self):
        if self.old_policy is None:
            return torch.tensor(0.0, device=self.cfg.device)
        kl = 0.0
        for k, v in self.policy.state_dict().items():
            if k in self.old_policy:
                diff = v - self.old_policy[k]
                kl += torch.mean(diff * diff)
        return kl
    
    def param_distance(self):
        if self.old_policy_params is None:
            return torch.tensor(0.0, device=self.cfg.device)
        dist = 0.0
        for p_new, p_old in zip(self.policy.parameters(), self.old_policy_params):
            dist += torch.sum((p_new - p_old) ** 2)
        return dist
    
    def _update_dap(self, reward_mean, uniqueness, step_idx):
       
        self.dap_reward_history.append(reward_mean)
        self.dap_uniq_history.append(uniqueness)
        
      
        if len(self.dap_reward_history) > self.dap_patience:
            self.dap_reward_history.pop(0)
            self.dap_uniq_history.pop(0)
        
        if len(self.dap_reward_history) < self.dap_patience:
            return  

        recent = np.mean(self.dap_reward_history[-self.dap_patience//2:])
        past = np.mean(self.dap_reward_history[:self.dap_patience//2])
        uniq_recent = np.mean(self.dap_uniq_history[-3:]) if len(self.dap_uniq_history) >= 3 else 1.0

        reward_stalled = recent <= past * 1.01
        low_uniqueness = uniq_recent < 0.7

        if reward_stalled or low_uniqueness:
            self.kl_lambda = max(self.kl_lambda_min, self.kl_lambda / self.dap_factor)
            mode = "explore"
            self.dap_last_explore = step_idx
        else:
            self.kl_lambda = min(self.kl_lambda_max, self.kl_lambda * self.dap_factor)
            mode = "exploit"

        self.writer.add_scalar("dap/kl_lambda", self.kl_lambda, step_idx)
        self.writer.add_scalar("dap/mode", 1.0 if mode == "explore" else 0.0, step_idx)

    def step(self, step_idx):
        device = self.cfg.device
        if step_idx % 5 == 0:
            self.backup_old_policy()

     
        oversample_factor = 1.3
        sample_size = min(int(round(self.cfg.batch_size * oversample_factor)), 1024)
        smiles_all, logp_all = self.policy.sample(sample_size, self.cfg.temperature, sampling=self.cfg.sampling)

        seen = set()
        smiles, logp_list = [], []
        for smi, lp in zip(smiles_all, logp_all):
            if smi and smi not in seen:
                seen.add(smi)
                smiles.append(smi)
                logp_list.append(lp)
            if len(smiles) >= self.cfg.batch_size:  
                break

        if len(smiles) == 0:
            print(f"[Warning] Step {step_idx}: no valid SMILES after dedup, skipping.")
            return {"step": step_idx, "reward_mean": 0.0, "reward_max": 0.0, "loss": 0.0}

        logp = torch.stack(logp_list).to(device)

        # vectors, _, is_valid, raw_props_list = multiobjective_vector_jnk3(
        #     smiles,
        #     props=list(self.weights.keys()),
        #     n_jobs=self.cfg.n_jobs,
        #     normalize=True,
        #     return_raw=True
        # )

        # vectors, _, is_valid, raw_props_list = multiobjective_vector(
        #     smiles,
        #     props=list(self.weights.keys()),
        #     n_jobs=self.cfg.n_jobs,
        #     normalize=True,
        #     return_raw=True
        # )

        if self.GPU_Docking:
            with MultiTargetDockingPoolGPU(
                target_pair=self.cfg.target_pair,          # 'gsk3b_jnk3' / 'dhodh_rorgt' /  'egfr_met'  /  pik3ca_mtor...
                sequential=True,) as pool:
                vectors, _, is_valid, raw_props_list = multiobjective_vector_dat_dual(
                    smiles,
                    props=list(self.weights.keys()),
                    pool=pool,
                    normalize=True,
                    return_raw=True,
                    admet=self.admet)

        elif self.admet:
            with DockingPool(self.receptor, self.center, self.size, n_procs=self.cfg.n_jobs, exhaustiveness=8) as pool:
                vectors, _, is_valid, raw_props_list = multiobjective_vector_dat_admet(
                    smiles,
                    props=list(self.weights.keys()),
                    pool=pool, normalize=True, return_raw=True, admet=self.admet,
                    admet_project=self.admet_project, admet_model=self.admet_model, admet_thr=self.admet_thr)
        elif self.dock:
            if self.receptor_dual is not None and self.center_dual is not None and self.size_dual is not None:
                targets = [
                    {"receptor": self.receptor, "center": self.center, "size": self.size},
                    {"receptor": self.receptor_dual, "center": self.center_dual, "size": self.size_dual},
                ]
                with MultiTargetDockingPool(targets, n_procs=self.cfg.n_jobs, exhaustiveness=8) as pool:
                    vectors, _, is_valid, raw_props_list = multiobjective_vector_dat_dual(
                        smiles,
                        props=list(self.weights.keys()),
                        pool=pool, normalize=True, return_raw=True, admet=self.admet
                    )
            else:
                with DockingPool(self.receptor, self.center, self.size, n_procs=self.cfg.n_jobs, exhaustiveness=8) as pool:
                    vectors, _, is_valid, raw_props_list = multiobjective_vector_dat(
                        smiles,
                        props=list(self.weights.keys()),
                        pool=pool, normalize=True, return_raw=True, admet=self.admet
                    )

            # else:              
            #      with DockingPool(self.receptor, self.center, self.size, n_procs=self.cfg.n_jobs, exhaustiveness=8) as pool:
            #         vectors, _, is_valid, raw_props_list = multiobjective_vector_dat(
            #             smiles,
            #             props=list(self.weights.keys()),
            #             pool=pool, normalize=True, return_raw=True, admet=self.admet)
        else:
            vectors, _, is_valid, raw_props_list = multiobjective_vector(
                smiles,
                props=list(self.weights.keys()),
                n_jobs=self.cfg.n_jobs,
                normalize=True,
                return_raw=True)

        is_valid = np.array(is_valid)

        if not np.any(is_valid):
            print(f"[Warning] Step {step_idx}: all invalid molecules, skipping update.")
            return {"step": step_idx, "reward_mean": 0.0, "reward_max": 0.0, "loss": 0.0}

        # === 4. Pareto & Crowding ===
        valid_indices = np.where(is_valid)[0]
        valid_vectors = vectors[valid_indices].tolist()

        ranks_full = np.full(len(smiles), len(smiles), dtype=np.float32)
        cd_full = np.zeros(len(smiles), dtype=np.float32)

        if len(valid_vectors) > 0:
            ranks_valid, fronts = pareto_sort(valid_vectors)
            cd_valid = crowding_distance(valid_vectors)
            ranks_full[valid_indices] = ranks_valid
            cd_full[valid_indices] = cd_valid

        ranks = torch.tensor(ranks_full, device=device, dtype=torch.float32)
        cd_all = torch.tensor(cd_full, device=device, dtype=torch.float32)

        rewards = torch.exp(-ranks) + self.crowd_alpha * (cd_all / (cd_all.max() + 1e-8))
        rewards = rewards * torch.tensor(is_valid, dtype=torch.float32, device=device)

        for i, smi in enumerate(smiles):
            if is_valid[i] and smi:
                self.smiles_freq[smi] += 1
                freq_penalty = 0.05 * np.log(1 + self.smiles_freq[smi])   # 频率惩罚


                # nov = self.mem.novelty(smi)
                # self.mem.add(smi)
                # base_reward = rewards[i]
                # diversity_reward = (1 - self.div_lambda) * base_reward + self.div_lambda * nov
                # rewards[i] = max(0.0, diversity_reward - freq_penalty)


                try:
                    fp_nov = self.mem.novelty(smi, metric="max")
                except:
                    fp_nov = 0.0
                
                if self.scaffold_weight > 0.0:
                     scaf_nov = self.scaffold_mem.novelty_score(smi)
                else:
                    scaf_nov = 0.0

                combined_nov = self.scaffold_weight * scaf_nov + (1.0 - self.scaffold_weight) * fp_nov
                self.mem.add(smi)
                base_reward = rewards[i]
                diversity_reward = (1 - self.div_lambda) * base_reward + self.div_lambda * combined_nov
                rewards[i] = max(0.0, diversity_reward - freq_penalty)
                # rewards[i] = (1 - self.div_lambda) * rewards[i] + self.div_lambda * nov

        if self.w_flag:
            self.adapt.update(raw_props_list)
            self.weights = self.adapt.weights
            for key in self.weights:
                self.weights[key] = min(0.99, self.weights[key])      # max
            if self.admet:
                #"hERG","AMES","DILI","CYP3A4_Veith","CYP2D6_Veith",
                # "HIA_Hou","Caco2_Wang","BBB_Martins"
                if self.safe_item:
                    self.weights['hERG'] = max(0.2, min(1.0, self.weights['hERG']))
                    self.weights['AMES'] = max(0.2, min(1.0, self.weights['AMES']))
                    self.weights['DILI'] = max(0.2, min(1.0, self.weights['DILI']))
                # self.weights['BBB_Martins'] = max(0.15, min(1.0, self.weights['BBB_Martins']))
                # 20260108
                # self.weights['Caco2_Wang'] = max(0.15, min(1.0, self.weights['BBB_Martins']))
                pass
           
            if 'dock' in self.weights:
                self.weights['dock'] = max(0.4, min(1.0, self.weights['dock']))
            if 'dock_0' in self.weights:
                self.weights['dock_0'] = max(0.4, min(1.0, self.weights['dock_0']))
            if 'dock_1' in self.weights:
                self.weights['dock_1'] = max(0.4, min(1.0, self.weights['dock_1']))
           
        
        torch.cuda.empty_cache()
        
        bl = self.baseline.update(rewards.mean().item())
        adv = rewards - bl
        param_dist = self.param_distance()
        loss = -(adv * logp.to(device)).mean() + self.kl_lambda * param_dist

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 2.0)
        self.optim.step()

        for r, s in zip(rewards.tolist(), smiles):
            self.topk.add((r, s))

        if (step_idx + 1) % self.cfg.save_every == 0:
            self.save(step_idx)
        
        

        if step_idx % 50 == 0:
            self.logger.log_frontier(step_idx, vectors, ranks.cpu().numpy(), smiles, raw_props_list)

        valid_smiles = [smiles[i] for i in range(len(smiles)) if is_valid[i]]
        if valid_smiles:
            uniqueness = len(set(valid_smiles)) / len(valid_smiles)
        else:
            uniqueness = 0.0



        row = {
            "step": step_idx,
            "loss": round(float(loss.item()), 5),
            "reward_mean": round(float(rewards.mean().item()), 5),
            "reward_max": round(float(rewards.max().item()), 5),
            "KL": round(float(param_dist.item()), 5),
            "baseline": round(float(bl), 5),
            "uniqueness": round(float(uniqueness), 5), 
        }  # 

        # if step_idx > 150:
        #     local_mean_reward = round(float(rewards.mean().item()), 5)
        #     if self.mean_reward < local_mean_reward:
        #         self.mean_reward = local_mean_reward
        #         torch.save(self.policy.vae.state_dict(), os.path.join(self.save_dir, f"policy_{step_idx + 1}.pt"))

       
        # if step_idx % 5 == 0:
        if self.dap_factor > 0.0:
            self._update_dap(
                    reward_mean=float(rewards.mean().item()),
                    uniqueness=float(uniqueness),
                    step_idx=step_idx
                )
        row["kl_lambda"] = round(float(self.kl_lambda), 5) 
        row["dap_factor"] = round(float(self.dap_factor), 5) 
        row["scaffold_weight"] = round(float(self.scaffold_weight), 5) # "dap_factor","scaffold_weight"

        for k, v in self.weights.items():
            row[f"w_{k}"] = round(float(v), 5)
        self.csv.log(row)

        # TensorBoard
        self.writer.add_scalar("loss", loss.item(), step_idx)
        self.writer.add_scalar("reward/mean", rewards.mean().item(), step_idx)
        self.writer.add_scalar("reward/max", rewards.max().item(), step_idx)
        self.writer.add_scalar("KL", param_dist.item(), step_idx)
        self.writer.add_scalar("metrics/uniqueness", uniqueness, step_idx)  
        for k, v in self.weights.items():
            self.writer.add_scalar(f"weights/{k}", v, step_idx)

        return row

    def save(self, step_idx):
        torch.save(self.policy.vae.state_dict(), os.path.join(self.save_dir, f"policy_{step_idx + 1}.pt"))
        with open(os.path.join(self.save_dir, f"topk_{step_idx + 1}.json"), "w", encoding="utf-8") as f:
            json.dump(self.topk.data, f, ensure_ascii=False, indent=2)
