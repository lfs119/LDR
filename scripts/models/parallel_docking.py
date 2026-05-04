import argparse
import csv
import sys
import os, time, json, math
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import atexit, weakref
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, QED, Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from contextlib import contextmanager
current_path = os.getcwd()
sys.path.append(os.path.join(current_path, 'scripts'))


from moses.metrics.SA_Score import sascorer

from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina


_G = {
    "vina": None,
    "vinas": None,
    "receptor": None,
    "box_center": None,
    "box_size": None,
    "exhaustiveness": 8,
    "cpu": 1,
    "pains": None,
    "meeko_preparer": None,
}

def _init_worker(receptor_pdbqt: str,
                 box_center: Tuple[float, float, float],
                 box_size: Tuple[float, float, float],
                 exhaustiveness: int = 8,
                 cpu: int = 1):
  
    _G["vina"] = Vina(sf_name="vina", cpu=max(1, int(cpu)), seed=os.getpid() & 0xFFFF, verbosity=0,)
    _G["vina"].set_receptor(receptor_pdbqt)
    _G["vina"].compute_vina_maps(center=list(box_center), box_size=list(box_size))
    _G["receptor"] = receptor_pdbqt
    _G["box_center"] = box_center
    _G["box_size"] = box_size
    _G["exhaustiveness"] = int(exhaustiveness)
    _G["cpu"] = int(cpu)
    if hasattr(_G["vina"], "set_verbosity"):
        _G["vina"].set_verbosity(0)

    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    _G["pains"] = FilterCatalog(params)

    _G["meeko_preparer"] = MoleculePreparation()  


def _ping() -> bool:
    return True


def _passes_quick_filters(mol: Chem.Mol,
                          pains: FilterCatalog,
                          min_mw: float = 150,
                          max_mw: float = 600,
                          max_rot: int = 12) -> bool:
    mw = Descriptors.MolWt(mol)
    if not (min_mw <= mw <= max_mw):
        return False
    if pains.HasMatch(mol):
        return False
    if Descriptors.NumRotatableBonds(mol) > max_rot:
        return False
    return True

def _smiles_to_rdkit3d(smiles: str) -> Optional[Chem.Mol]:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    m = Chem.AddHs(m)
    if AllChem.EmbedMolecule(m, AllChem.ETKDGv3()) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(m)
    return m

def vina_affinity_to_kd(aff_kcal: float, T: float = 298.15):
    R_kcal = 1.9872036e-3  # kcal/(mol·K)
    kd_M = math.exp(aff_kcal / (R_kcal * T))
    kd_M = float(np.clip(kd_M, 1e-15, 1e6))  
    pKd = -math.log10(kd_M)
    ki_nM = kd_M * 1e9
    return kd_M, pKd, ki_nM

def _to_M(x: float, unit: str = "M") -> float:
    unit = unit.strip().lower()
    scale = {"m":1.0, "mm":1e-3, "um":1e-6, "nm":1e-9, "pm":1e-12}
    if unit in ("m", "mol/l", "mol·l^-1", "mol/liter", "mol/litre"):
        return float(x)
    if unit in scale:
        return float(x) * scale[unit]
    raise ValueError(f"Unknown unit: {unit}")



def _normalize_dock_affinity(aff_kcal: float, aff_best=-15.0, aff_worst=-5.0) -> float:
    x = -aff_kcal
    lo, hi = -aff_worst, -aff_best
    t = (x - lo) / (hi - lo)
    return float(np.clip(t, 0.0, 1.0))

@contextmanager
def _silence_stdio():
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        with open(os.devnull, "wb") as dn:
            os.dup2(dn.fileno(), 1)
            os.dup2(dn.fileno(), 2)
        yield
    finally:
        os.dup2(saved_out, 1); os.close(saved_out)
        os.dup2(saved_err, 2); os.close(saved_err)

def _dock_affinity_from_smiles(smiles: str) -> Dict[str, Any]:
    """
    return:
      {'smiles','ok','aff','qed','sa','msg','kd_est_M','pKd_est','ki_est_nM'}
    """
    try:
        mol = _smiles_to_rdkit3d(smiles)
        if mol is None or mol.GetNumConformers() == 0:
            return {"smiles": smiles, "ok": False, "msg": "invalid_or_no_3d"}

        if not _passes_quick_filters(mol, _G["pains"]):
            return {"smiles": smiles, "ok": False, "msg": "failed_filters"}
        
        mol_NoH = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(mol_NoH)
        qed = float(QED.qed(mol_NoH))
        
        sa = float(sascorer.calculateScore(mol_NoH)) 
        logp = float(Descriptors.MolLogP(mol_NoH)) if mol_NoH is not None else 0.0

     
        prep = _G.get("meeko_preparer")
        if prep is None:
            prep = MoleculePreparation()
            _G["meeko_preparer"] = prep

        if hasattr(prep, "prepare"):
            setups = prep.prepare(mol)      # meeko 0.6.1: List[MoleculeSetup]
        else:
            setups = prep(mol)
        setup = setups[0] if isinstance(setups, (list, tuple)) else setups

    
        ws = PDBQTWriterLegacy.write_string(setup)
        if isinstance(ws, tuple):
            pdbqt_str, ok, msg = ws
            if not ok:
                return {"smiles": smiles, "ok": False, "msg": f"meeko_write:{msg}"}
        else:
            pdbqt_str = ws

        # v = _G["vina"]
        # v.set_ligand_from_string(pdbqt_str)
        # v.dock(exhaustiveness=_G["exhaustiveness"], n_poses=1)
        with _silence_stdio():
            v = _G["vina"]
            v.set_ligand_from_string(pdbqt_str)
            v.dock(exhaustiveness=_G["exhaustiveness"], n_poses=1)
        aff = float(v.energies()[0][0])  

     
        kd_M, pKd, ki_nM = vina_affinity_to_kd(aff)

        return {
            "smiles": smiles, "ok": True,
            "aff": aff, "qed": qed, "sa": sa, "msg": "ok",
            "kd_est_M": float(kd_M),
            "pKd_est": float(pKd),
            "ki_est_nM": float(ki_nM),
            "logp": float(logp),
        }

    except Exception as e:
        return {"smiles": smiles, "ok": False, "msg": f"exception:{e.__class__.__name__}"}


def linear_map(x, lo, hi):
    if hi == lo:
        return 0.0
    t = (x - lo) / (hi - lo)
    return float(np.clip(t, 0.0, 1.0))

def default_reward_shaping(aff_kcal: float,
                           qed: float,
                           sa: Optional[float] = None,
                           aff_best: float = -15.0,
                           aff_worst: float = -5.0,
                           w_aff: float = 0.7,
                           w_qed: float = 0.25,
                           w_sa: float = 0.05) -> float:
    dock_term = _normalize_dock_affinity(aff_kcal, aff_best=aff_best, aff_worst=aff_worst)
    sa_term = 1.0
    if sa is not None:
        sa_term = float(np.clip(1.0 - (sa / 10.0), 0.0, 1.0))  # SA: 1(易)~10(难)
    reward = w_aff * dock_term + w_qed * float(np.clip(qed, 0.0, 1.0)) + w_sa * sa_term
    return float(np.clip(reward, 0.0, 1.0))


_ACTIVE_POOLS = weakref.WeakSet()

class DockingPool:
    def __init__(self,
                 receptor_pdbqt: str,
                 box_center: Tuple[float, float, float],
                 box_size: Tuple[float, float, float],
                 n_procs: int = max(1, os.cpu_count() // 2),
                 exhaustiveness: int = 8,
                 threads_per_proc: int = 1):
        self.receptor = receptor_pdbqt
        self.center = tuple(map(float, box_center))
        self.size = tuple(map(float, box_size))
        self.n_procs = int(n_procs)
        self.exhaust = int(exhaustiveness)
        self.cpu = int(threads_per_proc)

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._executor = ProcessPoolExecutor(
            max_workers=self.n_procs,
            mp_context=mp.get_context("spawn"),  
            initializer=_init_worker,
            initargs=(self.receptor, self.center, self.size, self.exhaust, self.cpu)
        )
        self._closed = False

        fut = self._executor.submit(_ping)
        fut.result()

        _ACTIVE_POOLS.add(self)

        wr = weakref.ref(self)
        def _closer(wr=wr):
            obj = wr()
            if obj is not None:
                obj.shutdown()
        atexit.register(_closer)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()

    def score_batch(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        to_run = [s for s in smiles_list if s not in self._cache]
        futures = {self._executor.submit(_dock_affinity_from_smiles, s): s for s in to_run}
        for fut in as_completed(futures):
            res = fut.result()
            self._cache[res["smiles"]] = res
        return [self._cache[s] for s in smiles_list]

    def score_and_reward(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        raw = self.score_batch(smiles_list)
        out = []
        for r in raw:
            if r.get("ok"):
                rew = default_reward_shaping(r["aff"], r["qed"], r.get("sa"))
            else:
                rew = 0.0
            rr = dict(r)
            rr["reward"] = float(rew)
            out.append(rr)
        return out

    def shutdown(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
        self._executor = None  



def _close_all_pools():
    for p in list(_ACTIVE_POOLS):
        try:
            p.shutdown()
        except Exception:
            pass

_ACTIVE_POOLS = weakref.WeakSet()
atexit.register(_close_all_pools)


def calc_dat_properties(
    smiles: str,
    props=("dock", "qed", "sa", "aff"),
    *,
    pool: "DockingPool",
    on_error_default: float = 0.0,
    aff_best: float = -15.0,
    aff_worst: float = -5.0,
):
  
    out = {k: on_error_default for k in props}
    try:
        res = pool.score_batch([smiles])[0]
        if not res.get("ok", False):
            return out

        aff = float(res["aff"])
        qed = float(res.get("qed", 0.0))
        sa_raw = res.get("sa", None)

        dock = _normalize_dock_affinity(aff, aff_best=aff_best, aff_worst=aff_worst)
        sa_n = None
        if sa_raw is not None and np.isfinite(sa_raw):
            sa_n = float(np.clip(1.0 - (sa_raw / 10.0), 0.0, 1.0))

        if "dock" in props:      out["dock"] = dock
        if "qed" in props:       out["qed"]  = float(np.clip(qed, 0.0, 1.0))
        if "sa" in props and sa_n is not None: out["sa"] = sa_n
        if "aff" in props:       out["aff"]  = aff
        if "pKd_est" in props:   out["pKd_est"] = float(res.get("pKd_est", on_error_default))
        if "ki_est_nM" in props: out["ki_est_nM"] = float(res.get("ki_est_nM", on_error_default))
    except Exception:
        pass
    return out

def calc_dat_properties_batch(
    smiles_list: List[str],
    props=("dock", "qed", "sa", "aff", "logp"),
    add_props=(),
    *,
    pool: "DockingPool",
    on_error_default: float = 0.0,
    aff_best: float = -15.0,
    aff_worst: float = -5.0,
):
    outs = [{k: on_error_default for k in props} for _ in smiles_list]
    try:
        results = pool.score_batch(smiles_list)
        for i, res in enumerate(results):
            if not res.get("ok", False):
                continue
            aff = float(res["aff"])
            qed = float(res.get("qed", 0.0))
            sa_raw = res.get("sa", None)
            logp_raw = float(res.get("logp", 0.0))

            dock = _normalize_dock_affinity(aff, aff_best=aff_best, aff_worst=aff_worst)
            sa_n = None
            if sa_raw is not None and np.isfinite(sa_raw):
                sa_n = float(np.clip(1.0 - (sa_raw / 10.0), 0.0, 1.0))


            score = np.exp(-0.5 * ((logp_raw - 2.5) / max(1e-6, 1)) ** 2)
            logp = np.clip(score, 0.0, 1.0)

            o = outs[i]
            if "dock" in props:      o["dock"] = dock
            if "sa" in props and sa_n is not None: o["sa"] = sa_n
            if "logp" in props:      o["logp"] = float(logp)
            # if "pKd_est" in props:   o["pKd_est"] = float(res.get("pKd_est", on_error_default))
            # if "ki_est_nM" in props: o["ki_est_nM"] = float(res.get("ki_est_nM", on_error_default))
            if "qed" in props:       o["qed"]  = float(np.clip(qed, 0.0, 1.0))
            if "aff" in props:       o["aff"]  = aff
            if "sa_raw" in add_props: o["sa_raw"] = sa_raw
            if "logp_raw" in add_props: o["logp_raw"] = logp_raw
    except Exception:
        pass
    return outs

def calc_dat_reward(
    smiles: str,
    *,
    pool: "DockingPool",
    w_dock: float = 0.70,
    w_qed: float  = 0.25,
    w_sa: float   = 0.05,
    on_error_default: float = 0.0,
    aff_best: float = -15.0,
    aff_worst: float = -5.0,
):
    props = calc_dat_properties(
        smiles,
        props=("dock","qed","sa","aff","pKd_est","ki_est_nM"),
        pool=pool, on_error_default=on_error_default,
        aff_best=aff_best, aff_worst=aff_worst,
    )
    dock = float(props.get("dock", 0.0))
    qed  = float(props.get("qed", 0.0))
    sa   = float(props.get("sa",  0.0))
    reward = float(np.clip(w_dock * dock + w_qed * qed + w_sa * sa, 0.0, 1.0))
    props["reward"] = reward
    return props

def calc_dat_reward_batch(
    smiles_list: List[str],
    props=("dock", "qed", "sa", "aff", "pKd_est", "ki_est_nM"),
    *,
    pool: "DockingPool",
    w_dock: float = 0.70,
    w_qed: float  = 0.25,
    w_sa: float   = 0.05,
    on_error_default: float = 0.0,
    aff_best: float = -15.0,
    aff_worst: float = -5.0,
):
    props_list = calc_dat_properties_batch(
        smiles_list, props=props,
        pool=pool, on_error_default=on_error_default,
        aff_best=aff_best, aff_worst=aff_worst,
    )
    outs = []
    for p in props_list:
        dock = float(p.get("dock", 0.0))
        qed  = float(p.get("qed", 0.0))
        sa   = float(p.get("sa",  0.0))
        reward = float(np.clip(w_dock * dock + w_qed * qed + w_sa * sa, 0.0, 1.0))
        q = dict(p)
        q["reward"] = reward
        outs.append(q)
    return outs


def load_box(box_file: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    
    with open(box_file, "r") as f:
        txt = f.read().strip()
    try:
        obj = json.loads(txt)
        return tuple(obj["center"]), tuple(obj["size"])
    except Exception:
        cx = cy = cz = sx = sy = sz = None
        for line in txt.splitlines():
            line = line.strip().lower().replace(" ", "")
            for key in ["center_x", "center_y", "center_z", "size_x", "size_y", "size_z"]:
                if line.startswith(key):
                    val = float(line.split("=")[-1])
                    if key == "center_x": cx = val
                    elif key == "center_y": cy = val
                    elif key == "center_z": cz = val
                    elif key == "size_x": sx = val
                    elif key == "size_y": sy = val
                    elif key == "size_z": sz = val
        assert None not in (cx, cy, cz, sx, sy, sz), "box file parse failed"
        return (cx, cy, cz), (sx, sy, sz)


if __name__ == "__main__":
  
    parser = argparse.ArgumentParser(description= 'please enter paths')
    parser.add_argument('--receptor', type= str, default='/raid/home/xukai/FRATTVAE/data/hLRRK2.pdbqt', help= 'receptor.pdbqt')
    parser.add_argument('--box', type= str, default='/raid/home/xukai/FRATTVAE/data/hLRRK2.box.txt', help='ligand.box')
    parser.add_argument('--base_path', type= str, default= '/raid/home/xukai/FRATTVAE/results/CNS_SMILES_standardized_struct_1020/generate', help= 'base path')
    parser.add_argument('--csv_name', type= str, default= 'generate_frattvae_pro_lrrk2_20251025_104921_policy_250_20251026_211333_1.csv', help= 'csv name')
    args = parser.parse_args()

    base_path = args.base_path
    csv_name = args.csv_name
    receptor = args.receptor
    center, size = load_box(args.box)

    receptor_name = os.path.basename(receptor).split('.')[0]
    csv_path = os.path.join(base_path, csv_name)
    csv_cal_name = os.path.splitext(csv_name)[0] + '_' + receptor_name +'_vina.csv'
    csv_output_path = os.path.join(base_path, csv_cal_name)
    df = pd.read_csv(csv_path, encoding='utf-8')
    data = df['SMILES'].tolist()
    with DockingPool(receptor, center, size, n_procs=16, exhaustiveness=8, threads_per_proc=1) as pool:
        props_dicts = calc_dat_properties_batch(
            data, pool=pool,
            props=["dock", "aff"],
            add_props=()
        )
    
    original_columns = df.columns.tolist()
    original_cols_to_keep = ["SMILES", "MW", "NP", "TPSA", "QRCI", "QED", "SA", "logP"]
    missing_cols = [col for col in original_cols_to_keep if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in original CSV: {missing_cols}")
    
    if props_dicts:
         new_property_keys = list(props_dicts[0].keys())
    else:
         new_property_keys = ["dock", "sa", "logp", "qed", "aff"]

    final_fieldnames = original_cols_to_keep + new_property_keys 
    final_fieldnames = df.columns.tolist() + new_property_keys

    with open(csv_output_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=final_fieldnames)
        writer.writeheader()

        for i, (_, row) in enumerate(df.iterrows()):
            out_row = {col: row[col] for col in df.columns.tolist()}
            prop_dict = props_dicts[i] if i < len(props_dicts) else {}
            for key in new_property_keys:
                out_row[key] = prop_dict.get(key, None)
            writer.writerow(out_row)