import os, sys, math, json, atexit, weakref
from typing import List, Tuple, Dict, Any, Optional
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, QED, Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from moses.metrics.SA_Score import sascorer
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina


_G: Dict[str, Any] = {
    "vina": None,          
    "vinas": None,         
    "targets": None,        
    "exhaustiveness": 8,
    "cpu": 1,
    "pains": None,
    "meeko_preparer": None,
}


def _ping() -> bool:
    return True

def _passes_quick_filters(
    mol: Chem.Mol,
    pains: FilterCatalog,
    min_mw: float = 150,
    max_mw: float = 600,
    max_rot: int = 12
) -> bool:
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



def _init_common(exhaustiveness: int = 8, cpu: int = 1):
    _G["exhaustiveness"] = int(exhaustiveness)
    _G["cpu"] = int(cpu)

    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    _G["pains"] = FilterCatalog(params)

    _G["meeko_preparer"] = MoleculePreparation()


def _init_worker_single(
    receptor_pdbqt: str,
    box_center: Tuple[float, float, float],
    box_size: Tuple[float, float, float],
    exhaustiveness: int = 8,
    cpu: int = 1
):
   
    _init_common(exhaustiveness, cpu)

    v = Vina(sf_name="vina", cpu=max(1, int(cpu)), seed=os.getpid() & 0xFFFF, verbosity=0)
    v.set_receptor(receptor_pdbqt)
    v.compute_vina_maps(center=list(box_center), box_size=list(box_size))
    if hasattr(v, "set_verbosity"):
        v.set_verbosity(0)

    _G["vina"] = v
    _G["vinas"] = [v]
    _G["targets"] = [{"receptor": receptor_pdbqt, "center": box_center, "size": box_size}]


def _init_worker_multitarget(
    targets: List[Dict[str, Any]],
    exhaustiveness: int = 8,
    cpu: int = 1
):
    
    _init_common(exhaustiveness, cpu)

    vinas = []
    for t in targets:
        receptor = t["receptor"]
        center = t["center"]
        size = t["size"]
        v = Vina(sf_name="vina", cpu=max(1, int(cpu)), seed=os.getpid() & 0xFFFF, verbosity=0)
        v.set_receptor(receptor)
        v.compute_vina_maps(center=list(center), box_size=list(size))
        if hasattr(v, "set_verbosity"):
            v.set_verbosity(0)
        vinas.append(v)

    _G["vinas"] = vinas
    _G["vina"] = vinas[0] if vinas else None   
    _G["targets"] = targets


def _prepare_ligand_once(smiles: str) -> Dict[str, Any]:
    try:
        mol = _smiles_to_rdkit3d(smiles)
        if mol is None or mol.GetNumConformers() == 0:
            return {"smiles": smiles, "ok": False, "msg": "invalid_or_no_3d"}

        if not _passes_quick_filters(mol, _G["pains"]):
            return {"smiles": smiles, "ok": False, "msg": "failed_filters"}

        mol_NoH = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(mol_NoH)
        qed = float(QED.qed(mol_NoH))
        sa = float(sascorer.calculateScore(mol_NoH))  # 1~10（raw）
        logp = float(Descriptors.MolLogP(mol_NoH)) if mol_NoH is not None else 0.0

        prep = _G.get("meeko_preparer")
        if prep is None:
            prep = MoleculePreparation()
            _G["meeko_preparer"] = prep

        setups = prep.prepare(mol) if hasattr(prep, "prepare") else prep(mol)
        setup = setups[0] if isinstance(setups, (list, tuple)) else setups

        ws = PDBQTWriterLegacy.write_string(setup)
        if isinstance(ws, tuple):
            pdbqt_str, ok, msg = ws
            if not ok:
                return {"smiles": smiles, "ok": False, "msg": f"meeko_write:{msg}"}
        else:
            pdbqt_str = ws

        return {
            "smiles": smiles, "ok": True, "msg": "ok",
            "pdbqt_str": pdbqt_str,
            "qed": qed, "sa": sa, "logp": logp,
        }
    except Exception as e:
        return {"smiles": smiles, "ok": False, "msg": f"exception:{e.__class__.__name__}"}



def _dock_affinity_from_smiles_single(smiles: str) -> Dict[str, Any]:
    base = _prepare_ligand_once(smiles)
    if not base.get("ok", False):
        return base

    try:
        with _silence_stdio():
            v = _G["vina"]
            v.set_ligand_from_string(base["pdbqt_str"])
            v.dock(exhaustiveness=_G["exhaustiveness"], n_poses=1)

        aff = float(v.energies()[0][0])
        kd_M, pKd, ki_nM = vina_affinity_to_kd(aff)

        return {
            "smiles": smiles, "ok": True, "msg": "ok",
            "aff": aff,
            "qed": float(base["qed"]),
            "sa": float(base["sa"]),       # raw
            "logp": float(base["logp"]),   # raw
            "kd_est_M": float(kd_M),
            "pKd_est": float(pKd),
            "ki_est_nM": float(ki_nM),
        }
    except Exception as e:
        return {"smiles": smiles, "ok": False, "msg": f"exception:{e.__class__.__name__}"}


def _dock_affinity_from_smiles_multitarget(smiles: str) -> Dict[str, Any]:
  
    base = _prepare_ligand_once(smiles)
    if not base.get("ok", False):
        return base

    vinas: List[Vina] = _G.get("vinas") or []
    if not vinas:
        return {"smiles": smiles, "ok": False, "msg": "no_vinas_initialized"}

    per_target: List[Dict[str, Any]] = []
    affs: List[Optional[float]] = []

    for i, v in enumerate(vinas):
        try:
            with _silence_stdio():
                v.set_ligand_from_string(base["pdbqt_str"])
                v.dock(exhaustiveness=_G["exhaustiveness"], n_poses=1)
            aff = float(v.energies()[0][0])

            kd_M, pKd, ki_nM = vina_affinity_to_kd(aff)
            per_target.append({
                "ok": True,
                "aff": aff,
                "kd_est_M": float(kd_M),
                "pKd_est": float(pKd),
                "ki_est_nM": float(ki_nM),
            })
            affs.append(aff)
        except Exception as e:
            per_target.append({"ok": False, "msg": f"t{i}_exception:{e.__class__.__name__}"})
            affs.append(None)

    valid_affs = [a for a in affs if a is not None]
    ok_all = all((a is not None) for a in affs)
    ok_any = any((a is not None) for a in affs)

    if not valid_affs:
        return {"smiles": smiles, "ok": False, "msg": "failed_all_targets"}

    aff_best = float(min(valid_affs))  
    aff_worst = float(max(valid_affs)) 
    kd_M, pKd, ki_nM = vina_affinity_to_kd(aff_worst)

    out: Dict[str, Any] = {
        "smiles": smiles,
        "ok": bool(ok_all),                
        "ok_any": bool(ok_any),            
        "msg": "ok" if ok_all else "partial_ok",

        "qed": float(base["qed"]),
        "sa": float(base["sa"]),            # raw
        "logp": float(base["logp"]),        # raw

       
        "aff": float(aff_worst),
        "aff_best": aff_best,
        "aff_worst": aff_worst,

     
        "kd_est_M": float(kd_M),
        "pKd_est": float(pKd),
        "ki_est_nM": float(ki_nM),

      
        "targets": per_target,
        "n_targets": int(len(vinas)),
    }

    for i, a in enumerate(affs):
        out[f"aff_{i}"] = a
       

    for i, r in enumerate(per_target):
        if r.get("ok"):
            out[f"kd_est_M_{i}"] = r["kd_est_M"]
            out[f"pKd_est_{i}"] = r["pKd_est"]
            out[f"ki_est_nM_{i}"] = r["ki_est_nM"]
            # alias
            out[f"kd_est_M_t{i}"] = r["kd_est_M"]
            out[f"pKd_est_t{i}"] = r["pKd_est"]
            out[f"ki_est_nM_t{i}"] = r["ki_est_nM"]
        else:
            out[f"kd_est_M_{i}"] = None
            out[f"pKd_est_{i}"] = None
            out[f"ki_est_nM_{i}"] = None
            out[f"kd_est_M_t{i}"] = None
            out[f"pKd_est_t{i}"] = None
            out[f"ki_est_nM_t{i}"] = None

    return out


_ACTIVE_POOLS = weakref.WeakSet()

def _close_all_pools():
    for p in list(_ACTIVE_POOLS):
        try:
            p.shutdown()
        except Exception:
            pass

atexit.register(_close_all_pools)


class DockingPool:
    def __init__(
        self,
        receptor_pdbqt: str,
        box_center: Tuple[float, float, float],
        box_size: Tuple[float, float, float],
        n_procs: int = max(1, os.cpu_count() // 2),
        exhaustiveness: int = 8,
        threads_per_proc: int = 1
    ):
        self.receptor = receptor_pdbqt
        self.center = tuple(map(float, box_center))
        self.size = tuple(map(float, box_size))
        self.n_procs = int(n_procs)
        self.exhaust = int(exhaustiveness)
        self.cpu = int(threads_per_proc)

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._closed = False

        self._executor = ProcessPoolExecutor(
            max_workers=self.n_procs,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker_single,
            initargs=(self.receptor, self.center, self.size, self.exhaust, self.cpu),
        )

        # 拉起进程确保 initializer 已执行
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
        futures = {self._executor.submit(_dock_affinity_from_smiles_single, s): s for s in to_run}
        for fut in as_completed(futures):
            res = fut.result()
            self._cache[res["smiles"]] = res
        return [self._cache[s] for s in smiles_list]

    def shutdown(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
        self._executor = None


class MultiTargetDockingPool:
  
    def __init__(
        self,
        targets: List[Dict[str, Any]],
        n_procs: int = max(1, os.cpu_count() // 2),
        exhaustiveness: int = 8,
        threads_per_proc: int = 1
    ):
        if not targets or len(targets) < 1:
            raise ValueError("targets must be a non-empty list")

        # 规范化 targets
        norm_targets = []
        for t in targets:
            norm_targets.append({
                "receptor": t["receptor"],
                "center": tuple(map(float, t["center"])),
                "size": tuple(map(float, t["size"])),
            })

        self.targets = norm_targets
        self.n_procs = int(n_procs)
        self.exhaust = int(exhaustiveness)
        self.cpu = int(threads_per_proc)

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._closed = False

        self._executor = ProcessPoolExecutor(
            max_workers=self.n_procs,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker_multitarget,
            initargs=(self.targets, self.exhaust, self.cpu),
        )

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
        futures = {self._executor.submit(_dock_affinity_from_smiles_multitarget, s): s for s in to_run}
        for fut in as_completed(futures):
            res = fut.result()
            self._cache[res["smiles"]] = res
        return [self._cache[s] for s in smiles_list]

    def shutdown(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
        self._executor = None



def calc_dat_properties_batch_extended(
    smiles_list: List[str],
    props=("dock", "qed", "sa", "aff", "logp"),
    add_props=(),
    *,
    pool,
    on_error_default: float = 0.0,
    aff_best: float = -15.0,
    aff_worst: float = -5.0,
):
    outs = [{k: on_error_default for k in props} for _ in smiles_list]
    try:
        results = pool.score_batch(smiles_list)
        for i, res in enumerate(results):
            if not res.get("ok", False) and not res.get("ok_any", False):
                continue

            o = outs[i]

            aff = res.get("aff", None)
            qed = float(res.get("qed", 0.0))
            sa_raw = res.get("sa", None)
            logp_raw = float(res.get("logp", 0.0))

          
            if aff is not None:
                dock = _normalize_dock_affinity(float(aff), aff_best=aff_best, aff_worst=aff_worst)
            else:
                dock = 0.0

          
            sa_n = None
            if sa_raw is not None and np.isfinite(sa_raw):
                sa_n = float(np.clip(1.0 - (float(sa_raw) / 10.0), 0.0, 1.0))

          
            score = np.exp(-0.5 * ((logp_raw - 2.5) / max(1e-6, 1)) ** 2)
            logp = float(np.clip(score, 0.0, 1.0))

         
            if "dock" in props: o["dock"] = float(dock)
            if "qed" in props:  o["qed"] = float(np.clip(qed, 0.0, 1.0))
            if "sa" in props and sa_n is not None: o["sa"] = float(sa_n)
            if "logp" in props: o["logp"] = float(logp)
            if "aff" in props and aff is not None: o["aff"] = float(aff)

           
            for k in props:
                if k in ("dock", "qed", "sa", "logp", "aff"):
                    continue
                if k in res:
                    o[k] = res[k]

          
            for k in props:
                if k.startswith("dock_") or k.startswith("dock_t"):
                    suffix = k.split("_", 1)[1]  # "0" or "t0"
                    aff_key = f"aff_{suffix}" if suffix.isdigit() else f"aff_{suffix[1:]}"  # t0 -> aff_0
                    a = res.get(aff_key, None)
                    if a is not None:
                        o[k] = _normalize_dock_affinity(float(a), aff_best=aff_best, aff_worst=aff_worst)
                    else:
                        o[k] = on_error_default

         
            if "sa_raw" in add_props:
                o["sa_raw"] = sa_raw
            if "logp_raw" in add_props:
                o["logp_raw"] = logp_raw

    except Exception:
        pass
    return outs


if __name__ == "__main__":
   
    receptor0 = "/raid/home/xukai/FRATTVAE/data/hjnk3.pdbqt"
    center0, size0 = load_box("/raid/home/xukai/FRATTVAE/data/hjnk3.box.txt")

    receptor1 = "/raid/home/xukai/FRATTVAE/data/hgsk3b.pdbqt"
    center1, size1 = load_box("/raid/home/xukai/FRATTVAE/data/hgsk3b.box.txt")


    smiles = ["O=C1c2ccccc2-c2nccc3ccnc1c23", "Cc1nc(C(=S)SC(=O)Cc2ccc(C=CC#N)cc2)cs1", "C=C(Cc1ccc(N2CCC(C)(N)CC2)cc1)C(CCc1cc(Cl)ccc1C(=O)O)(OC)c1ccccc1"]
    # with DockingPool(receptor0, center0, size0, n_procs=4, exhaustiveness=8) as pool:
    #     print(pool.score_batch(smiles))


    targets = [
        {"receptor": receptor0, "center": center0, "size": size0},
        {"receptor": receptor1, "center": center1, "size": size1},
    ]
    with MultiTargetDockingPool(targets, n_procs=4, exhaustiveness=8) as pool:
        res = pool.score_batch(smiles)
        print(res)
    pass
