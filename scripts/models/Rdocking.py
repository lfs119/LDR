from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdFMCS
from meeko import PDBQTMolecule, RDKitMolCreate
import math

def rdkit_mols_from_vina_pdbqt(pdbqt_file: str):
    """读取 Vina 的 .pdbqt，返回 RDKit 分子列表（每个pose一个conformer或多个mol）"""
    # Vina 的文件：不需要 is_dlg=True
    pdbqt_mol = PDBQTMolecule.from_file(pdbqt_file, skip_typing=True)
    rdkit_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)  # list[Chem.Mol]
    # 有的版本会把多个 pose 放到同一个 Mol 的不同 conformer里，也可能拆成多个 Mol
    return rdkit_list

def mol_from_any(path: str) -> Chem.Mol:
    ext = path.lower().split(".")[-1]
    if ext == "sdf":
        sup = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
        return sup[0]
    if ext == "mol2":
        return Chem.MolFromMol2File(path, removeHs=False, sanitize=False)
    if ext == "mol":
        return Chem.MolFromMolFile(path, removeHs=False, sanitize=False)
    if ext == "pdb":
        return Chem.MolFromPDBFile(path, removeHs=False, sanitize=False)
    if ext == "pdbqt":
        # 若是字符串而不是文件，用 PDBQTMolecule(STRING, ...) 构造也可以
        pdbqt_mol = PDBQTMolecule.from_file(path, skip_typing=True)
        return RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)[0]
    raise ValueError("Unsupported format")

def best_rmsd(m1: Chem.Mol, m2: Chem.Mol) -> float:
    """两分子 重原子RMSD； 自动处理原子顺序不同问题"""
    a = Chem.RemoveHs(m1); b = Chem.RemoveHs(m2)
    try:
        if a.GetNumAtoms() == b.GetNumAtoms():
            return float(rdMolAlign.GetBestRMS(a, b))
    except Exception:
        pass
    try:
        match = a.GetSubstructMatch(b)
        if match and len(match) == b.GetNumAtoms():
            amap = list(zip(match, range(b.GetNumAtoms())))
            return float(rdMolAlign.AlignMol(a, b, atomMap=amap))
    except Exception:
        pass
    try:
        res = rdFMCS.FindMCS([a, b], completeRingsOnly=True, ringMatchesRingOnly=True, timeout=10)
        if res.numAtoms > 0:
            patt = Chem.MolFromSmarts(res.smartsString)
            a_idx = a.GetSubstructMatch(patt); b_idx = b.GetSubstructMatch(patt)
            if a_idx and b_idx and len(a_idx) == len(b_idx):
                amap = list(zip(a_idx, b_idx))
                return float(rdMolAlign.AlignMol(a, b, atomMap=amap))
    except Exception:
        pass
    return math.nan

def rmsd_vina_pdbqt_vs_crystal_sdf(vina_pdbqt: str, crystal_sdf: str, pose_idx: int = 0) -> float:
    """从 Vina 的 .pdbqt 取第 pose_idx 个姿势，与晶体 SDF 计算（重原子）RMSD"""
    rdkit_list = rdkit_mols_from_vina_pdbqt(vina_pdbqt)
    if not rdkit_list:
        return math.nan
    dock_mol = rdkit_list[min(pose_idx, len(rdkit_list)-1)]
    xtal_mol = mol_from_any(crystal_sdf)
    return best_rmsd(dock_mol, xtal_mol)


if __name__ == "__main__":
    # from argparse import ArgumentParser
    # parser = ArgumentParser()
    # parser.add_argument("--pdbqt", type=str, default="/raid/home/xukai/FRATTVAE/frame_9999_apo_out.pdbqt")
    # parser.add_argument("--sdf", type=str, default='/raid/home/xukai/FRATTVAE/data/7NVH_AVO.sdf')
    # args = parser.parse_args()
    pdbqt_path = "/raid/home/xukai/FRATTVAE/scripts/frame_9999_apo_out.pdbqt"
    sdf_path = "/raid/home/xukai/FRATTVAE/data/7NVH_AVO.sdf"
    # r = rmsd_pdbqt_vs_sdf(args.pdbqt, args.sdf, pose_idx=0)
    r = rmsd_vina_pdbqt_vs_crystal_sdf(pdbqt_path, sdf_path, pose_idx=0)
    print("Top-1 RMSD =", r, "Å")


    import numpy as np
    from sklearn.metrics import r2_score          # 或直接用 scipy.stats.linregress

    x = np.array([42.0, 44.0, 43.2, 45.2, 45.6, 46.5])
    y = np.array([-9.113, -8.979, -8.975, -8.859, -8.646, -8.347])

    slope, intercept = np.polyfit(x, y, 1)        # 一次多项式拟合
    y_pred = slope * x + intercept

    r2 = r2_score(y, y_pred)                     # 决定系数
    print(f"slope = {slope:.4f}, intercept = {intercept:.4f}, R² = {r2:.4f}")


