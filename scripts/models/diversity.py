import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

class DiversityMemory:
    def __init__(self, fp_size=2048, radius=2, max_size=5000, strategy="fifo"):
        """
        Parameters:
            fp_size (int): Morgan fingerprint bit length.
            radius (int): Morgan fingerprint radius (e.g., 2 for ECFP4).
            max_size (int): Maximum number of stored fingerprints.
            strategy (str): "fifo" or "random" for eviction.
        """
        self.fp_size = fp_size
        self.radius = radius
        self.max_size = max_size
        self.strategy = strategy
        self.fps = []          # list of fingerprints
        self.smiles_set = set()  # optional: avoid duplicate SMILES

    def _fp(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.radius, nBits=self.fp_size, useChirality=True
            )
            return fp
        except Exception:
            return None

    def add(self, smi):
        if smi in self.smiles_set:
            return  # avoid duplicate storage
        fp = self._fp(smi)
        if fp is None:
            return

        self.smiles_set.add(smi)
        if len(self.fps) >= self.max_size:
            if self.strategy == "fifo":
                self.fps.pop(0)
                # Note: smiles_set not pruned → memory leak
                # For strict memory control, use a deque with (smi, fp)
            elif self.strategy == "random":
                idx = np.random.randint(len(self.fps))
                self.fps.pop(idx)
        self.fps.append(fp)

    def get_scaffold(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold, isomericSmiles=False)
        except:
            return ""

    def novelty(self, smi, metric="max"):
        """
        Compute novelty of a SMILES string w.r.t. stored fingerprints.
        
        Parameters:
            smi (str): Input SMILES.
            metric (str): "max", "mean", "top3", or "entropy".
        
        Returns:
            float: Novelty score in [0, 1]. Higher = more novel.
        """
        if not self.fps:
            return 1.0

        fp = self._fp(smi)
        if fp is None:
            return 0.0  # invalid molecule → not novel

        # Compute Tanimoto similarities
        sims = []
        for f in self.fps:
            try:
                sim = DataStructs.TanimotoSimilarity(fp, f)
                sims.append(sim)
            except Exception:
                sims.append(0.0)

        sims = np.array(sims)

        if metric == "max":
            return 1.0 - float(np.max(sims))
        elif metric == "mean":
            return 1.0 - float(np.mean(sims))
        elif metric == "top3":
            top3 = np.mean(np.partition(sims, -3)[-3:])  # average of top-3 similarities
            return 1.0 - float(top3)
        elif metric == "entropy":
            # Higher entropy = more uniform similarity → possibly more novel?
            # Not standard; use with caution
            from scipy.stats import entropy
            # Bin similarities
            hist, _ = np.histogram(sims, bins=10, range=(0,1), density=True)
            hist = np.clip(hist, 1e-8, None)
            ent = entropy(hist, base=2)
            return ent / np.log2(10)  # normalize to [0,1]
        else:
            raise ValueError(f"Unknown metric: {metric}")