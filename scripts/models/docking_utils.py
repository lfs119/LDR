from __future__ import annotations
from functools import cache
import os
import os
from pathlib import Path
from typing import Callable, List, Dict, Tuple
# import os
# import tempfile
import subprocess
# from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem import AllChem
import hashlib

from meeko import MoleculePreparation, PDBQTWriterLegacy


####### SET YOUR DOCKING PATH PREFIX #######
DOCKING_PATH_PREFIX = Path("/home/xukai_cluster/FRATTVAE_800/scripts/models/docking")


def _stable_name(smiles: str) -> str:
    """Stable, filesystem-safe id (avoid python hash randomness / negative names)."""
    return hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:16]


def _pdbqt_is_degenerate(p: Path, tol: float = 1e-6) -> bool:
    """Reject ligands with essentially collapsed coordinates (prevents Vina axis_frame nrm~0)."""
    try:
        coords = []
        for line in p.read_text(errors="ignore").splitlines():
            if line.startswith(("ATOM", "HETATM")):
                # PDBQT uses PDB-like columns; coordinates are at 30:38,38:46,46:54
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except Exception:
                    continue
                coords.append((x, y, z))
        if len(coords) < 2:
            return True

        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]

        if (max(xs) - min(xs) < tol) and (max(ys) - min(ys) < tol) and (max(zs) - min(zs) < tol):
            return True

        # also guard against too-few unique coordinates
        uniq = set((round(x, 6), round(y, 6), round(z, 6)) for x, y, z in coords)
        if len(uniq) < max(2, len(coords) // 10):
            return True

        return False
    except Exception:
        return True


def _prepare_single_ligand_meeko(args: Tuple[str, Path, Path]) -> Tuple[str, Path | None]:
    """Prepare a single ligand for docking using RDKit + Meeko.

    Args:
        args: Tuple of (smiles, ligand_dir, tmp_path)

    Returns:
        Tuple of (smiles, pdbqt_path) if successful, (smiles, None) if failed
    """
    smiles, ligand_dir, tmp_path = args  # keep signature
    name = hash(smiles)

    pdbqt_path = ligand_dir / f"{name}.pdbqt"

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, None
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
            return smiles, None
        AllChem.MMFFOptimizeMolecule(mol)
  

        prep = MoleculePreparation()
        molsetup = prep.prepare(mol)
        molsetup = molsetup[0] if isinstance(molsetup, (list, tuple)) else molsetup

        # str 或 (str, ok, msg)
        ret = PDBQTWriterLegacy.write_string(molsetup)
        
        if isinstance(ret, tuple):
            pdbqt_str, ok, msg = ret[0], ret[1], ret[2] if len(ret) > 2 else ""
            if not ok:
                # 生成失败
                return smiles, None
        else:
             pdbqt_str = ret
             
        pdbqt_path.write_text(pdbqt_str)  

        # 5) Guard: reject degenerate coordinates that crash Vina-GPU
        if _pdbqt_is_degenerate(pdbqt_path):
            pdbqt_path.unlink(missing_ok=True)
            return smiles, None

        return smiles, pdbqt_path

    except Exception:
        pdbqt_path.unlink(missing_ok=True)
        return smiles, None


def _prepare_ligands_meeko(smiles_list: List[str], tmp_path: Path, n_proc: int = 12) -> Dict[str, Path]:
    """Prepare ligands for docking by converting SMILES to PDBQT files in parallel.

    Args:
        smiles_list: List of SMILES strings to prepare
        tmp_path: Path to temporary directory for files

    Returns:
        Dictionary mapping SMILES to their PDBQT file paths
    """
    ligand_dir = tmp_path / "ligands"
    ligand_dir.mkdir(exist_ok=True)

    args = [(smiles, ligand_dir, tmp_path) for smiles in smiles_list]

    smiles_to_pdbqt: Dict[str, Path] = {}

    # NOTE: for VS Code debugpy, n_proc>1 can be unstable; for training it's fine.
    with Pool(processes=n_proc) as pool:
        for smiles, pdbqt_path in pool.imap(_prepare_single_ligand_meeko, args, chunksize=8):
            if pdbqt_path is not None:
                smiles_to_pdbqt[smiles] = pdbqt_path
    
    # for smiles_path in ligand_dir.glob('*.smiles'):
    #     smiles_path.unlink()
        
    # No .smiles temp files in this version, so nothing to clean
    return smiles_to_pdbqt

def _prepare_single_ligand(args: Tuple[str, Path, Path]) -> Tuple[str, Path | None]:
    """Prepare a single ligand for docking.
    
    Args:
        args: Tuple of (smiles, ligand_dir, tmp_path)
        
    Returns:
        Tuple of (smiles, pdbqt_path) if successful, (smiles, None) if failed
    """
    smiles, ligand_dir, tmp_path = args
    name = hash(smiles)
    smiles_path = ligand_dir / f"{name}.smiles"
    mol2_path = ligand_dir / f"{name}.mol2"
    pdb_path = ligand_dir / f"{name}.pdb"
    pdbqt_path = ligand_dir / f"{name}.pdbqt"
    
    # Write SMILES file
    with open(smiles_path, 'w') as f:
        f.write(smiles)
    
    try:
        # SMILES to mol2 with 3D coordinates
        result = subprocess.run(['obabel', 
                               str(smiles_path), 
                               '-O', str(mol2_path),
                               '--gen3d', 'best',
                               '-p', '7.4'],
                              capture_output=True,
                              text=True)
        
        if result.returncode != 0:
            print(f"Failed to convert SMILES to mol2 for {smiles}: {result.stderr}")
            return smiles, None
            
        # mol2 to PDB
        result = subprocess.run(['obabel',
                               str(mol2_path),
                               '-O', str(pdb_path),
                               '-h',
                               '--gen3d', 'best',
                               '-p', '7.4'],  # Add hydrogens
                              capture_output=True,
                              text=True)
                              
        if result.returncode != 0:
            print(f"Failed to convert mol2 to PDB for {smiles}: {result.stderr}")
            return smiles, None
            
        # PDB to PDBQT
        result = subprocess.run(['obabel',
                               str(pdb_path),
                               '-O', str(pdbqt_path),
                               '--gen3d', 'best',
                               '-p', '7.4',
                               '--partialcharge', 'gasteiger'],
                              capture_output=True,
                              text=True)
                              
        if result.returncode != 0:
            print(f"Failed to convert PDB to PDBQT for {smiles}: {result.stderr}")
            return smiles, None
            
        # clean intermediate files
        smiles_path.unlink(missing_ok=True)
        mol2_path.unlink(missing_ok=True)
        pdb_path.unlink(missing_ok=True)
        
        return smiles, pdbqt_path
        
    except Exception as e:
        print(f"Conversion failed for {smiles}: {str(e)}")
        smiles_path.unlink(missing_ok=True)
        mol2_path.unlink(missing_ok=True)
        pdb_path.unlink(missing_ok=True)
        pdbqt_path.unlink(missing_ok=True)
        return smiles, None



def _prepare_ligands(smiles_list: List[str], tmp_path: Path, n_proc: int = 24) -> Dict[str, Path]:
    """Prepare ligands for docking by converting SMILES to PDBQT files in parallel.
    
    Args:
        smiles_list: List of SMILES strings to prepare
        tmp_path: Path to temporary directory for files
        
    Returns:
        Dictionary mapping SMILES to their PDBQT file paths
    """
    ligand_dir = tmp_path / "ligands"
    ligand_dir.mkdir(exist_ok=True)
    
    args = [(smiles, ligand_dir, tmp_path) for smiles in smiles_list]
    
    smiles_to_pdbqt = {}
    with Pool(processes=n_proc) as pool:
        for smiles, pdbqt_path in pool.imap(_prepare_single_ligand, args):
            if pdbqt_path is not None:
                smiles_to_pdbqt[smiles] = pdbqt_path
    
    for smiles_path in ligand_dir.glob('*.smiles'):
        smiles_path.unlink()
        
    return smiles_to_pdbqt

def _run_docking(smiles_to_pdbqt: Dict[str, Path], 
                receptor_path: str,
                task_id: str,
                center: Tuple[float, float, float],
                tmp_path: Path) -> Dict[str, float]:
    """Run docking for prepared ligands against a receptor.
    
    Args:
        smiles_to_pdbqt: Dictionary mapping SMILES to their PDBQT file paths
        receptor_path: Path to receptor PDBQT file
        task_id: for temporary directories in tmp_path
        center: (x, y, z) coordinates of binding site center
        tmp_path: Path to temporary directory
        
    Returns:
        Dictionary mapping SMILES to their docking scores
    """
    # task-specific directories and config files
    task_dir = tmp_path / f"task_{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    output_dir = task_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    DOCKING_PATH_PREFIX = Path("/home/xukai_cluster")
    
    
    config_path = task_dir / "config.txt"
    with open(config_path, 'w') as f:
        f.write(f"""receptor = {receptor_path}
center_x = {center[0]}
center_y = {center[1]}
center_z = {center[2]}
size_x = 20
size_y = 20
size_z = 20
thread = 1000
num_modes = 2
rilc_bfgs = 1
ligand_directory = {(tmp_path / "ligands").absolute()}
output_directory = {output_dir.absolute()}
opencl_binary_path = {DOCKING_PATH_PREFIX}/Vina-GPU-2.1/QuickVina2-GPU-2.1""")
    
    """ # Debug: Print ligand directory contents
    print("\nLigand directory contents:")
    for file in (tmp_path / "ligands").glob('*'):
        print(f"  {file.name}") """
    
    # print(DOCKING_PATH_PREFIX)
    # Run QuickVina2-GPU 
    # /home/xukai_cluster/Vina-GPU-2.1/QuickVina2-GPU-2.1
    vina_dir = DOCKING_PATH_PREFIX / "Vina-GPU-2.1" / "QuickVina2-GPU-2.1"
    result = subprocess.run(['./QuickVina2-GPU-2-1', '--config', str(config_path.absolute())], 
                          cwd=vina_dir,
                          capture_output=True,
                          text=True)
    
    if result.returncode != 0:
       print(f"\n++++++++++++++QuickVina2-GPU failed++++++++++++++++++++:")
       print(f"stdout: {result.stdout}")
       print(f"stderr: {result.stderr}")
        
    scores = {}
    for smiles in smiles_to_pdbqt:
        output_path = output_dir / f"{hash(smiles)}_out.pdbqt"
        try:
            with open(output_path) as f:
                for line in f:
                    if "REMARK VINA RESULT" in line:
                        score = float(line.split()[3])
                        scores[smiles] = score
                        break
        except:
            scores[smiles] = 0.0 
            
    return scores


