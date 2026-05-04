import argparse
import copy
import os
import random
import sys
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

import moses
# from molvs import
def clearAtomMapNums(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

def standardize_and_metrics(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        mol = copy.deepcopy(mol)
        clearAtomMapNums(mol)


        # mol = Chem.AddHs(mol)
        # AllChem.MMFFOptimizeMolecule(mol)




        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
    
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)

        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)

        # Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        smi_std = Chem.MolToSmiles(mol)
        # 计算描述符
        natom = mol.GetNumAtoms()
        mw = moses.metrics.weight(mol)
        logP = moses.metrics.logP(mol)
        qed = moses.metrics.QED(mol)
        sa = moses.metrics.SA(mol)
        npl = moses.metrics.NP(mol)
        tpsa = Chem.Descriptors.TPSA(mol)
        ct = Descriptors.BertzCT(mol)
    except:
        mol = smi_std = None
        natom = mw = logP = qed = sa = npl = tpsa = ct =  None

    if smi_std is None:
        print('none', flush= True)
    
    return smi, smi_std, natom, mw, logP, qed, sa, npl, tpsa, ct

if __name__ == "__main__":
    smiles_batch = ["COC(O)[C@H]1[C@@H](OC(O)C2CCCCC2)C[C@@H]2CC[C@H]1N2C", 
                    "C[C@@H]1CN(c2cc(-c3n[nH]c4ccc(OC5(C)CC5)cc34)ncn2)C[C@H](C)O1",
                    "CC1(C)O[C@@H](O)C2CCC(NC3NCC(C4NNCO4)C(N[C@H](CO)C4CCCCC4)N3)CC21"]
    
    smiles_batch = ["COC(O)[C@H]1[C@@H](OC(O)C2CCCCC2)C[C@@H]2CC[C@H]1N2C"]
    for i, smi in enumerate(smiles_batch):
       smi, smi_std, natom, mw, logP, qed, sa, npl, tpsa, ct = standardize_and_metrics(smi)
       print(f"smi = {smi}, smi_std = {smi_std}, qed = {qed}, sa= {sa}, logp = {logP} " )

# export PYTHONPATH="/home/xukai/xk/FRATTVAE/scripts:$PYTHONPATH"

# smi = COC(O)[C@H]1[C@@H](OC(O)C2CCCCC2)C[C@@H]2CC[C@H]1N2C, smi_std = COC(O)[C@H]1[C@@H](OC(O)C2CCCCC2)C[C@@H]2CC[C@H]1N2C, qed = 0.756840083558848, sa= 5.261368931532977, logp = 1.7177999999999993 
# smi = C[C@@H]1CN(c2cc(-c3n[nH]c4ccc(OC5(C)CC5)cc34)ncn2)C[C@H](C)O1, smi_std = C[C@@H]1CN(c2cc(-c3n[nH]c4ccc(OC5(C)CC5)cc34)ncn2)C[C@H](C)O1, qed = 0.7468766326209479, sa= 3.667179474274503, logp = 3.5648000000000026 
# smi = CC1(C)O[C@@H](O)C2CCC(NC3NCC(C4NNCO4)C(N[C@H](CO)C4CCCCC4)N3)CC21, smi_std = CC1(C)O[C@@H](O)C2CCC(NC3NCC(C4NNCO4)C(N[C@H](CO)C4CCCCC4)N3)CC21, qed = 0.2477917056232887, sa= 5.789092552391619, logp = -0.15439999999999499 