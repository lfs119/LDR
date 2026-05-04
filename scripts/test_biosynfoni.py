from biosynfoni import Biosynfoni
from rdkit import Chem

smi = 'COc1c(NC(=O)c2ccc(CC3C(=O)N(C)C(=O)N3C)cc2)ccnc1C'
mol = Chem.MolFromSmiles(smi)
fp = Biosynfoni(mol).fingerprint  # returns biosynfoni's count fingerprint of the molecule
print(len(fp))
smi = 'CC1(C)CC(c2nc(Cc3ccccn3)nn2C(=O)N2CC3(CCN(C4CCCCC4O)CC3)CC2=O)CCO1'
mol = Chem.MolFromSmiles(smi)
fp = Biosynfoni(mol).fingerprint
print(len(fp))

smi = 'CC(=O)c1ccc(N2CCN(Cc3noc(-c4ncc5c(n4)-c4ccccc4CC5)n3)CC2)nc1'
mol = Chem.MolFromSmiles(smi)
fp = Biosynfoni(mol).fingerprint
print(len(fp))