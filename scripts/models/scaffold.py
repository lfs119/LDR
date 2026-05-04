from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem

class ScaffoldMemory:
    def __init__(self):
        self.seen_scaffolds = set()

    def _get_scaffold(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        try:
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            # Canonical SMILES without chirality for consistency
            scaffold_smi = Chem.MolToSmiles(
                scaffold_mol, isomericSmiles=False, canonical=True
            )
            return scaffold_smi
        except Exception:
            return None

    def add(self, smi):
        scaf = self._get_scaffold(smi)
        if scaf:
            self.seen_scaffolds.add(scaf)

    def is_novel(self, smi):
        scaf = self._get_scaffold(smi)
        if scaf is None:
            return False
        return scaf not in self.seen_scaffolds

    def novelty_score(self, smi):
        return 1.0 if self.is_novel(smi) else 0.0