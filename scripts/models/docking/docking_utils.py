from __future__ import annotations
from functools import cache
from pathlib import Path
from typing import Callable, List, Dict, Tuple
# import os
# import tempfile
import subprocess
# from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool



####### SET YOUR DOCKING PATH PREFIX #######
DOCKING_PATH_PREFIX = Path("/home/xukai_cluster/FRATTVAE_800/scripts/models/docking")
############################################


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

def _prepare_ligands(smiles_list: List[str], tmp_path: Path, n_proc: int = 48) -> Dict[str, Path]:
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
thread = 5000
num_modes = 5
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
       print(f"\nQuickVina2-GPU failed:")
    #    print(f"stdout: {result.stdout}")
    #    print(f"stderr: {result.stderr}")
        
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


