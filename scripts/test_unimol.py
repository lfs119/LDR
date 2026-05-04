import numpy as np
from unimol_tools import UniMolRepr
# single smiles unimol representation
clf = UniMolRepr(data_type='molecule', 
                 remove_hs=False,
                 model_name='unimolv1', # avaliable: unimolv1, unimolv2
                 model_size='84m', # work when model_name is unimolv2. avaliable: 84m, 164m, 310m, 570m, 1.1B.
                 )
smiles = 'c1ccc(cc1)C2=NCC(=O)Nc3c2cc(cc3)[N+](=O)[O]'
smiles_list = [smiles]
unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs=True)
# CLS token repr
print(np.array(unimol_repr['cls_repr']).shape)
# atomic level repr, align with rdkit mol.GetAtoms()
print(np.array(unimol_repr['atomic_reprs']).shape)