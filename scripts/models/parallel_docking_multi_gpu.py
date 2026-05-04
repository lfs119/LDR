import os
import math
import atexit
import weakref
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from moses.metrics.SA_Score import sascorer

from .docking_utils import _prepare_ligands, _run_docking, _prepare_ligands_meeko


DEFAULT_DOCKING_PATH_PREFIX = Path("/home/xukai_cluster/FRATTVAE_800/scripts/models/docking")

def build_tasks(target_pair: str, prefix: Path, tmp_path: Path, smiles_to_pdbqt: Dict[str, str]):
    if target_pair == "gsk3b_jnk3":
        return [
            dict(smiles_to_pdbqt=smiles_to_pdbqt,
                 receptor_path=str(prefix / "6Y9S.pdbqt"),
                 task_id="gsk3b",
                 center=(24.503, 9.183, 9.226),
                 tmp_path=tmp_path),
            dict(smiles_to_pdbqt=smiles_to_pdbqt,
                 receptor_path=str(prefix / "4WHZ.pdbqt"),
                 task_id="jnk3",
                 center=(4.327, 101.902, 141.338),
                 tmp_path=tmp_path),
        ]
    elif target_pair == "dhodh_rorgt":
        return [
            dict(smiles_to_pdbqt=smiles_to_pdbqt,
                 receptor_path=str(prefix / "6QU7.pdbqt"),
                 task_id="dhodh",
                 center=(33.359, -11.558, -22.820),
                 tmp_path=tmp_path),
            dict(smiles_to_pdbqt=smiles_to_pdbqt,
                 receptor_path=str(prefix / "5NTP.pdbqt"),
                 task_id="rorgt",
                 center=(18.003, 11.762, 20.391),
                 tmp_path=tmp_path),
        ]
    elif target_pair == "egfr_met":
        return [
            dict(smiles_to_pdbqt=smiles_to_pdbqt,
                 receptor_path=str(prefix / "1M17.pdbqt"),
                 task_id="egfr",
                 center=(22.014, 0.253, 52.79),
                 tmp_path=tmp_path),
            dict(smiles_to_pdbqt=smiles_to_pdbqt,
                 receptor_path=str(prefix / "4MXC.pdbqt"),
                 task_id="met",
                 center=(-9.384, 17.423, -28.886),
                 tmp_path=tmp_path),
        ]
    elif target_pair == "pik3ca_mtor":
        return [
            dict(smiles_to_pdbqt=smiles_to_pdbqt,
                 receptor_path=str(prefix / "8V8I.pdbqt"),
                 task_id="pik3ca",
                 center=(-19.947, -23.175, 10.569),
                 tmp_path=tmp_path),
            dict(smiles_to_pdbqt=smiles_to_pdbqt,
                 receptor_path=str(prefix / "3FAP.pdbqt"),
                 task_id="mtor",
                 center=(-8.630, 26.528, 36.52),
                 tmp_path=tmp_path),
        ]
    else:
        raise ValueError(f"Invalid target_pair: {target_pair}")


def vina_affinity_to_kd(aff_kcal: float, T: float = 298.15):
    R_kcal = 1.9872036e-3
    kd_M = math.exp(aff_kcal / (R_kcal * T))
    kd_M = float(np.clip(kd_M, 1e-15, 1e6))
    pKd = -math.log10(kd_M)
    ki_nM = kd_M * 1e9
    return kd_M, pKd, ki_nM


def _basic_props(smiles: str) -> Dict[str, float]:
    """返回 raw qed/sa/logp（sa: 1~10）"""
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return {"qed": 0.0, "sa": 10.0, "logp": 0.0}
        Chem.SanitizeMol(m)
        return {
            "qed": float(QED.qed(m)),
            "sa": float(sascorer.calculateScore(m)),
            "logp": float(Descriptors.MolLogP(m)),
        }
    except Exception:
        return {"qed": 0.0, "sa": 10.0, "logp": 0.0}


_ACTIVE_POOLS = weakref.WeakSet()

def _close_all_pools():
    for p in list(_ACTIVE_POOLS):
        try:
            p.shutdown()
        except Exception:
            pass

atexit.register(_close_all_pools)


class MultiTargetDockingPoolGPU:
  
    def __init__(
        self,
        target_pair: str,
        *,
        docking_prefix: Path = DEFAULT_DOCKING_PATH_PREFIX,
        n_procs: int = 4,              
        sequential: bool = False,      
        tmp_root: str = "./tmp",
        cuda_visible_devices: Optional[str] = None,
        zero_is_failure: bool = True,  
    ):
        self.target_pair = target_pair
        self.prefix = Path(docking_prefix)
        self.n_procs = int(n_procs)
        self.sequential = bool(sequential)
        self.tmp_root = tmp_root
        self.zero_is_failure = bool(zero_is_failure)

        os.makedirs(self.tmp_root, exist_ok=True)
        # if cuda_visible_devices is not None:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._closed = False

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

    def shutdown(self):
        self._closed = True

    def score_batch(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        to_run = [s for s in smiles_list if s not in self._cache]

        if to_run:
            with tempfile.TemporaryDirectory(dir=self.tmp_root, prefix="gpu_pair_") as tmp_dir:
                tmp_path = Path(tmp_dir)
                
                smiles_to_pdbqt = _prepare_ligands_meeko(to_run, tmp_path) 

                tasks = build_tasks(self.target_pair, self.prefix, tmp_path, smiles_to_pdbqt)

                results = [None, None]
                
                # env = os.environ.copy()


                if self.sequential:
                    for i, task in enumerate(tasks):
                        try:
                            results[i] = _run_docking(
                                task["smiles_to_pdbqt"],
                                task["receptor_path"],
                                task["task_id"],
                                task["center"],
                                task["tmp_path"],
                            )
                        except Exception:
                            results[i] = {sm: 0.0 for sm in smiles_to_pdbqt}
                else:
                    with ThreadPoolExecutor(max_workers=2) as ex:
                        futs = {
                            ex.submit(
                                _run_docking,
                                task["smiles_to_pdbqt"],
                                task["receptor_path"],
                                task["task_id"],
                                task["center"],
                                task["tmp_path"],
                            ): i
                            for i, task in enumerate(tasks)
                        }
                        for fut in as_completed(futs):
                            i = futs[fut]
                            try:
                                results[i] = fut.result()
                            except Exception:
                                results[i] = {sm: 0.0 for sm in smiles_to_pdbqt}

                ds1_scores = results[0] or {}
                ds2_scores = results[1] or {}

                for sm in to_run:
                    base = _basic_props(sm)

                    if sm not in smiles_to_pdbqt:
                        self._cache[sm] = {
                            "smiles": sm, "ok": False, "ok_any": False, "msg": "prep_failed",
                            **base, "n_targets": 2, "targets": [{"ok": False}, {"ok": False}],
                            "aff": 0.0, "aff_0": 0.0, "aff_1": 0.0,
                        }
                        continue

                    a0 = ds1_scores.get(sm, 0.0)
                    a1 = ds2_scores.get(sm, 0.0)
                    try: a0 = float(a0)
                    except Exception: a0 = 0.0
                    try: a1 = float(a1)
                    except Exception: a1 = 0.0

                    def _is_valid_aff(a: float) -> bool:
                        if not np.isfinite(a):
                            return False
                        if self.zero_is_failure and abs(a) < 1e-12:
                            return False
                        return True

                    ok0 = _is_valid_aff(a0)
                    ok1 = _is_valid_aff(a1)
                    ok_any = ok0 or ok1
                    ok_all = ok0 and ok1

                    targets_out = []
                    affs_valid = []

                    if ok0:
                        kd_M, pKd, ki_nM = vina_affinity_to_kd(a0)
                        targets_out.append({"ok": True, "aff": a0, "kd_est_M": kd_M, "pKd_est": pKd, "ki_est_nM": ki_nM})
                        affs_valid.append(a0)
                    else:
                        targets_out.append({"ok": False, "aff": 0.0})

                    if ok1:
                        kd_M, pKd, ki_nM = vina_affinity_to_kd(a1)
                        targets_out.append({"ok": True, "aff": a1, "kd_est_M": kd_M, "pKd_est": pKd, "ki_est_nM": ki_nM})
                        affs_valid.append(a1)
                    else:
                        targets_out.append({"ok": False, "aff": 0.0})

                    if not ok_any:
                        self._cache[sm] = {
                            "smiles": sm,
                            "ok": False,
                            "ok_any": False,
                            "msg": "dock_failed_all_targets",
                            **base,
                            "n_targets": 2,
                            "targets": targets_out,
                            "aff": 0.0, "aff_0": 0.0, "aff_1": 0.0,
                        }
                        continue

                    aff_best = float(min(affs_valid))
                    aff_worst = float(max(affs_valid))
                    kd_M, pKd, ki_nM = vina_affinity_to_kd(aff_worst)
                    
                    mol_NoH = Chem.MolFromSmiles(sm)
                    Chem.SanitizeMol(mol_NoH)
                    qed = float(QED.qed(mol_NoH))
                    
                    sa = float(sascorer.calculateScore(mol_NoH))  # MOSES SA: 1(易)~10(难)
                    logp = float(Descriptors.MolLogP(mol_NoH)) if mol_NoH is not None else 0.0
                    

                    out = {
                        "smiles": sm,
                        "qed": qed, "sa": sa, "logp": logp,
                        "ok": bool(ok_all),
                        "ok_any": bool(ok_any),
                        "msg": "ok" if ok_all else "partial_ok",
                        **base,
                        "n_targets": 2,
                        "targets": targets_out,

                        "aff_0": float(a0) if ok0 else 0.0,
                        "aff_1": float(a1) if ok1 else 0.0,

                        "aff_best": float(aff_best),
                        "aff_worst": float(aff_worst),
                        "aff": float(aff_worst), 

                        "kd_est_M": float(kd_M),
                        "pKd_est": float(pKd),
                        "ki_est_nM": float(ki_nM),
                    }

                    if ok0:
                        out["kd_est_M_0"] = targets_out[0]["kd_est_M"]
                        out["pKd_est_0"] = targets_out[0]["pKd_est"]
                        out["ki_est_nM_0"] = targets_out[0]["ki_est_nM"]
                    else:
                        out["kd_est_M_0"] = None; out["pKd_est_0"] = None; out["ki_est_nM_0"] = None

                    if ok1:
                        out["kd_est_M_1"] = targets_out[1]["kd_est_M"]
                        out["pKd_est_1"] = targets_out[1]["pKd_est"]
                        out["ki_est_nM_1"] = targets_out[1]["ki_est_nM"]
                    else:
                        out["kd_est_M_1"] = None; out["pKd_est_1"] = None; out["ki_est_nM_1"] = None

                    self._cache[sm] = out

        return [self._cache.get(s, {"smiles": s, "ok": False, "msg": "missing"}) for s in smiles_list]
    
    

