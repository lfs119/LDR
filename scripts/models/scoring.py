# ====================== src/scoring.py ======================
import csv
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
import numpy as np
import sys, os

current_path = os.getcwd() 
sys.path.append(os.path.join(current_path, 'scripts'))
from models.parallel_docking import DockingPool
from moses.metrics.SA_Score import sascorer
from moses.metrics.NP_Score import npscorer
from moses.metrics.utils import qeppi, qrci
from joblib import Parallel, delayed
from collections import defaultdict
import threading, math, json, os


_normalizer_lock = threading.Lock()
_running_stats = defaultdict(lambda: {"mean": 0.0, "M2": 0.0, "n": 0})


def _calc_properties(
    smiles: str,
    props=("qed", "sa", "logp", "qeppi", "np", "qrci"),
    *,
    logp_mu: float = 2.5,
    logp_sigma: float = 1.0,
    np_mode: str = "sigmoid",         # 可选: "sigmoid" 或 "minmax"
    np_minmax: tuple = (-5.0, 5.0),
    qrci_minmax: tuple = (-1.0, 5.0),
    on_error_default: float = 0.0,
):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: on_error_default for k in props}

    def _safe(v):
        if not np.isfinite(v):
            return on_error_default
        return float(v)

    out = {k: on_error_default for k in props}
    try:
        Chem.SanitizeMol(mol)
        if "qed" in props:
            out["qed"] = np.clip(_safe(QED.qed(mol)), 0.0, 1.0)

        if "sa" in props:
            sa_v = _safe(sascorer.calculateScore(mol) / 10.0)
            out["sa"] = np.clip(1.0 - sa_v, 0.0, 1.0)

        if "logp" in props:
            lp = _safe(Descriptors.MolLogP(mol))
            score = np.exp(-0.5 * ((lp - logp_mu) / max(1e-6, logp_sigma)) ** 2)
            out["logp"] = np.clip(score, 0.0, 1.0)

        if "qeppi" in props:
            out["qeppi"] = np.clip(_safe(qeppi(smiles)), 0.0, 1.0)

        if "np" in props:
            np_raw = _safe(npscorer.scoreMol(mol))
            if np_mode == "sigmoid":
                np_val = 1.0 / (1.0 + np.exp(-np.clip(np_raw, -10, 10)))
            else:
                lo, hi = np_minmax
                np_val = (np_raw - lo) / (hi - lo + 1e-8)
            out["np"] = np.clip(np_val, 0.0, 1.0)

        if "qrci" in props:
            qr_raw = _safe(qrci(smiles))
            lo, hi = qrci_minmax
            qr_val = (qr_raw - lo) / (hi - lo + 1e-8)
            out["qrci"] = np.clip(qr_val, 0.0, 1.0)

    except Exception:
        pass
    return out

def _calc_raw_properties(
    smiles: str,
    props=("qed", "sa", "logp", "qeppi", "np", "qrci"),
    *,
    on_error_default: float = 0.0,
):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: on_error_default for k in props}

    def _safe(v):
        if not np.isfinite(v):
            return on_error_default
        return float(v)

    out = {k: on_error_default for k in props}
    try:
        Chem.SanitizeMol(mol)
        if "qed" in props:
            out["qed"] = _safe(QED.qed(mol))

        if "sa" in props:
            out["sa"] = _safe(sascorer.calculateScore(mol))

        if "logp" in props:
            out["logp"] = _safe(Descriptors.MolLogP(mol))

        if "qeppi" in props:
            out["qeppi"] = _safe(qeppi(smiles))

        if "np" in props:
            out["np"] = _safe(npscorer.scoreMol(mol))

        if "qrci" in props:
            out["qrci"] = _safe(qrci(smiles))

    except Exception:
        pass

    return out


def _invert_normalized_properties(
    norm_vals: dict,
    props=("qed", "sa", "logp", "qeppi", "np", "qrci"),
    *,
    logp_mu: float = 2.5,
    logp_sigma: float = 1.0,
    np_mode: str = "sigmoid",
    np_minmax: tuple = (-5.0, 5.0),
    qrci_minmax: tuple = (-1.0, 5.0),
    on_error_default: float = 0.0,
):
    """
    尝试从 _calc_properties 的归一化输出中恢复原始属性值。
    """
    raw = {}

    for p in props:
        val = norm_vals.get(p, on_error_default)
        if not (0.0 <= val <= 1.0):
            raw[p] = on_error_default
            continue

        try:
            if p == "qed":
               
                raw["qed"] = float(val)

            elif p == "sa":
                # sa_norm = clip(1.0 - sa_raw/10, 0, 1)
                # => sa_raw = (1.0 - sa_norm) * 10
                sa_norm = val
                sa_raw = (1.0 - sa_norm) * 10.0
            
                raw["sa"] = sa_raw

            elif p == "logp":
                # norm = exp(-0.5 * ((lp - mu)/sigma)^2)
                # => ((lp - mu)/sigma)^2 = -2 * ln(norm)
                # => lp = mu ± sigma * sqrt(-2 * ln(norm))
                if val <= 0.0:
                    raw["logp"] = on_error_default
                elif val >= 1.0:
                    # 最大概率点：lp = mu
                    raw["logp"] = logp_mu
                else:
                    z_sq = -2.0 * math.log(val)
                    if z_sq < 0:
                        raw["logp"] = logp_mu
                    else:
                        z = math.sqrt(z_sq)
                       
                        lp1 = logp_mu + logp_sigma * z
                        lp2 = logp_mu - logp_sigma * z
                        raw["logp"] = lp2  

            elif p == "qeppi":
              
                raw["qeppi"] = float(val)

            elif p == "np":
                if np_mode == "sigmoid":
                    # np_norm = sigmoid(np_raw) = 1 / (1 + exp(-np_raw))
                    # => np_raw = -log(1/np_norm - 1)
                    if val <= 0.0 or val >= 1.0:
                        raw["np"] = -10.0 if val <= 0.0 else 10.0
                    else:
                        raw["np"] = -math.log(1.0 / val - 1.0)
                else:
                    # min-max: np_norm = (np_raw - lo) / (hi - lo)
                    lo, hi = np_minmax
                    np_raw = val * (hi - lo) + lo
                    raw["np"] = np_raw

            elif p == "qrci":
                lo, hi = qrci_minmax
                qr_raw = val * (hi - lo) + lo
                raw["qrci"] = qr_raw

            else:
                raw[p] = on_error_default

        except (ValueError, OverflowError, ZeroDivisionError):
            raw[p] = on_error_default

    return raw

def _update_running_stats(values_dict):
    with _normalizer_lock:
        for k, v in values_dict.items():
            st = _running_stats[k]
            st["n"] += 1
            delta = v - st["mean"]
            st["mean"] += delta / st["n"]
            st["M2"] += delta * (v - st["mean"])
            _running_stats[k] = st


def _normalize_property_vector(vector, keys):
    normed = []
    with _normalizer_lock:
        for i, k in enumerate(keys):
            st = _running_stats[k]
            var = st["M2"] / max(1, st["n"] - 1)
            std = max(1e-6, math.sqrt(var))
            mean = st["mean"]
            val = (vector[i] - mean) / std
            normed.append(float(np.clip(val, -5.0, 5.0)))
    return normed


def multiobjective_vector(
    smiles_list,
    props=("qed", "sa", "logp", "qeppi", "np", "qrci"),
    n_jobs: int = 8,
    normalize: bool = True,
    return_raw: bool = True,
):
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_calc_properties)(s, props) for s in smiles_list
    )

    vectors, valid_smiles = [], []
    is_valid = [r is not None for r in results]

    for i, r in enumerate(results):
        if r is not None:
            _update_running_stats(r)
            vec = [r[k] for k in props]
            if normalize and all(_running_stats[k]["n"] > 10 for k in props):
                vec = _normalize_property_vector(vec, props)
        else:
            vec = [0.0] * len(props)
        vectors.append(vec)
        valid_smiles.append(smiles_list[i])  

    vectors = np.array(vectors, dtype=np.float32)

    if return_raw:
        return vectors, valid_smiles, is_valid, results
    else:
        return vectors, valid_smiles, is_valid

def _normalize_dock_affinity(aff_kcal: float, aff_best=-15.0, aff_worst=-5.0) -> float:
    x = -aff_kcal
    lo, hi = -aff_worst, -aff_best
    t = (x - lo) / (hi - lo)
    return float(np.clip(t, 0.0, 1.0))

def multiobjective_vector_dat(
    smiles_list,
    props=("dock", "qed", "sa", "aff", "pKd_est", "ki_est_nM", "logp"),
    *,
    pool: DockingPool,
    normalize: bool = True,
    return_raw: bool = True,
    logp_mu: float = 2.5,
    logp_sigma: float = 1.0,
    admet: bool = False,
):
 
  
    results = pool.score_batch(smiles_list)  

  
    vectors, valid_smiles, is_valid = [], [], []
    raw_props = []

    for s, r in zip(smiles_list, results):
        if r is not None and r.get("ok", False):
            aff = float(r.get("aff", 0.0))
            qed = float(np.clip(r.get("qed", 0.0), 0.0, 1.0))
            sa_raw = r.get("sa", None)
            dock = _normalize_dock_affinity(aff) 
            sa_n = float(np.clip(1.0 - (sa_raw / 10.0), 0.0, 1.0)) if sa_raw is not None else 0.0

            mol = Chem.MolFromSmiles(s)
            logp = float(Descriptors.MolLogP(mol)) if mol is not None else 0.0
            score = np.exp(-0.5 * ((logp - logp_mu) / max(1e-6, logp_sigma)) ** 2)
            logp = np.clip(score, 0.0, 1.0)

          
            sample = {
                "dock": dock,
                "qed": qed,
                "sa": sa_n,
                "aff": aff,
                # "pKd_est": float(r.get("pKd_est", 0.0)),
                # "ki_est_nM": float(r.get("ki_est_nM", 0.0)),
                "logp": logp,
            }
           
            for k in props:
                sample.setdefault(k, 0.0)

            _update_running_stats({k: sample[k] for k in props})
            vec = [sample[k] for k in props]
            if normalize and all(_running_stats[k]["n"] > 10 for k in props):
                vec = _normalize_property_vector(vec, props)

            vectors.append(vec)
            raw_props.append(sample)
            is_valid.append(True)
            valid_smiles.append(s)
        else:
           
            sample = {
                "dock": 0.0,
                "qed":  0.0,
                "sa":  0.0,
                "aff":  0.0,
                # "pKd_est": float(r.get("pKd_est", 0.0)),
                # "ki_est_nM": float(r.get("ki_est_nM", 0.0)),
                "logp":  0.0,
            }
            vectors.append([0.0]*len(props))
            raw_props.append(sample)
            is_valid.append(False)
            valid_smiles.append(s)

    vectors = np.array(vectors, dtype=np.float32)
    if return_raw:
        return vectors, valid_smiles, is_valid, raw_props
    else:
        return vectors, valid_smiles, is_valid


import re
_aff_idx_pat = re.compile(r"^aff_(\d+)$")

def _infer_n_targets_from_result(r: dict) -> int:
   
    if r is None:
        return 0
    n = r.get("n_targets", None)
    if isinstance(n, (int, np.integer)) and n > 0:
        return int(n)
 
    idxs = []
    for k in r.keys():
        m = _aff_idx_pat.match(k)
        if m:
            idxs.append(int(m.group(1)))
    if idxs:
        return max(idxs) + 1
  
    t = r.get("targets", None)
    if isinstance(t, list) and len(t) > 0:
        return len(t)
    return 0


def multiobjective_vector_dat_dual(
    smiles_list,
    props=("dock", "qed", "sa", "aff", "pKd_est", "ki_est_nM", "logp"),
    *,
    pool,                      
    normalize: bool = True,
    return_raw: bool = True,
    logp_mu: float = 2.5,
    logp_sigma: float = 1.0,
    admet: bool = False,
    allow_partial: bool = False, 
):
   
    results = pool.score_batch(smiles_list)

    vectors, valid_smiles, is_valid = [], [], []
    raw_props = []

    for s, r in zip(smiles_list, results):
        ok = False
        if r is not None:
            ok = bool(r.get("ok", False))
            if (not ok) and allow_partial:
                ok = bool(r.get("ok_any", False))

        if ok:
            aff = r.get("aff", 0.0)
            try:
                aff = float(aff)
            except Exception:
                aff = 0.0

            qed = float(np.clip(r.get("qed", 0.0), 0.0, 1.0))
            sa_raw = r.get("sa", None)
            try:
                sa_raw = float(sa_raw) if sa_raw is not None else None
            except Exception:
                sa_raw = None

            dock = _normalize_dock_affinity(aff)

            sa_n = 0.0
            if sa_raw is not None and np.isfinite(sa_raw):
                sa_n = float(np.clip(1.0 - (sa_raw / 10.0), 0.0, 1.0))

            logp_raw = r.get("logp", None)
            if logp_raw is None:
                mol = Chem.MolFromSmiles(s)
                logp_raw = float(Descriptors.MolLogP(mol)) if mol is not None else 0.0
            else:
                try:
                    logp_raw = float(logp_raw)
                except Exception:
                    logp_raw = 0.0
            score = np.exp(-0.5 * ((logp_raw - logp_mu) / max(1e-6, logp_sigma)) ** 2)
            logp = float(np.clip(score, 0.0, 1.0))

            n_targets = _infer_n_targets_from_result(r)
            dock_min = dock
            aff_best = aff
            aff_worst = aff

            if n_targets >= 2:
                aff_list = []
                dock_list = []
                for i in range(n_targets):
                    a_i = r.get(f"aff_{i}", None)
                    try:
                        a_i = float(a_i) if a_i is not None else None
                    except Exception:
                        a_i = None
                    aff_list.append(a_i)
                    if a_i is None:
                        dock_list.append(0.0)
                    else:
                        dock_list.append(_normalize_dock_affinity(a_i))

                valid_affs = [a for a in aff_list if a is not None]
                valid_docks = [d for (a, d) in zip(aff_list, dock_list) if a is not None]

                if valid_affs:
                    aff_best = float(min(valid_affs))   
                    aff_worst = float(max(valid_affs))  
                if valid_docks:
                    dock_min = float(min(valid_docks))

                dock = dock_min

            sample = {
                "dock": float(dock),
                "qed": float(qed),
                "sa": float(sa_n),
                "aff": float(aff),
                "logp": float(logp),

                "dock_min": float(dock_min),
                "aff_best": float(aff_best),
                "aff_worst": float(aff_worst),
            }

            if n_targets >= 2:
                for i in range(n_targets):
                    a_i = r.get(f"aff_{i}", None)
                    try:
                        a_i = float(a_i) if a_i is not None else 0.0
                    except Exception:
                        a_i = 0.0
                    sample[f"aff_{i}"] = a_i
                    sample[f"dock_{i}"] = float(_normalize_dock_affinity(a_i)) if a_i != 0.0 else 0.0

            for k in props:
                if k in sample:
                    continue
                if r is not None and (k in r):
                    v = r.get(k, 0.0)
                    try:
                        sample[k] = float(v)
                    except Exception:
                        sample[k] = 0.0
                else:
                    sample[k] = 0.0

         
            for k in props:
                if k.startswith("dock_t"):
                    try:
                        idx = int(k[6:])
                        a_i = r.get(f"aff_{idx}", None)
                        a_i = float(a_i) if a_i is not None else None
                        sample[k] = _normalize_dock_affinity(a_i) if a_i is not None else 0.0
                    except Exception:
                        sample[k] = 0.0

           
            for k in props:
                sample.setdefault(k, 0.0)

            _update_running_stats({k: sample[k] for k in props})

            vec = [sample[k] for k in props]
            if normalize and all(_running_stats[k]["n"] > 10 for k in props):
                vec = _normalize_property_vector(vec, props)

            vectors.append(vec)
            raw_props.append(sample)
            is_valid.append(True)
            valid_smiles.append(s)

        else:
          
            sample = {k: 0.0 for k in props}
            vectors.append([0.0] * len(props))
            raw_props.append(sample)
            is_valid.append(False)
            valid_smiles.append(s)

    vectors = np.array(vectors, dtype=np.float32)
    if return_raw:
        return vectors, valid_smiles, is_valid, raw_props
    else:
        return vectors, valid_smiles, is_valid

def _clamp01(x): 
    try:
        return float(min(1.0, max(0.0, x)))
    except:
        return 0.0

def _percentile_score(row, base_key, invert=False):
    k = f"{base_key}_drugbank_approved_percentile"
    if k in row:
        v = _clamp01(row[k] / 100.0)
        return 1.0 - v if invert else v
    return None  


_PROB_LOWER_BETTER = {
    "hERG", "AMES", "DILI", "ClinTox", "Carcinogens_Lagunin",
    "CYP1A2_Veith", "CYP2C19_Veith", "CYP2C9_Veith", "CYP2D6_Veith", "CYP3A4_Veith",
   
    "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53",
    "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase","NR-ER","NR-ER-LBD","NR-PPAR-gamma",
    "Skin_Reaction",
}


_PROB_HIGHER_BETTER = {
    "HIA_Hou","BBB_Martins","Bioavailability_Ma","PAMPA_NCATS",
}


_REG_RANGE = {
    "Caco2_Wang": (0.0, 1.0),     
    "Solubility_AqSolDB": (-12.0, 0.0),  # logS
    "VDss_Lombardo": (-1.0, 2.0),
    "PPBR_AZ": (0.0, 1.0),
    "Lipophilicity_AstraZeneca": (0.0, 6.0),
    "Half_Life_Obach": (0.0, 48.0),
    "Clearance_Hepatocyte_AZ": (0.0, 100.0),
    "Clearance_Microsome_AZ": (0.0, 100.0),
    "LD50_Zhu": (0.0, 10.0),       
}

def _scale_linear(x, lo, hi, invert=False):
    if hi <= lo: 
        return 0.0
    v = (x - lo) / (hi - lo)
    v = _clamp01(v)
    return 1.0 - v if invert else v

def _score_admet_row(row_dict, keys, project="oral_peripheral"):
    out = {}
    is_cns = ("cns" in str(project).lower())
    for k in keys:
        
        if k.endswith("_drugbank_approved_percentile") and k in row_dict:
            out[k] = _clamp01(row_dict[k] / 100.0)
            continue

      
        pct = _percentile_score(row_dict, k, invert=False)
        if pct is not None:
           
            if k in _PROB_LOWER_BETTER:
                pct = 1.0 - pct
           
            if k == "BBB_Martins" and not is_cns:
                pct = 1.0 - pct
            if k == "Pgp_Broccatelli" and is_cns:
                pct = 1.0 - pct
            out[k] = _clamp01(pct)
            continue

       
        if k in row_dict and isinstance(row_dict[k], (int, float)):
            v = float(row_dict[k])
            if k in _PROB_LOWER_BETTER:
                out[k] = _clamp01(1.0 - v)
                continue
            if k in _PROB_HIGHER_BETTER or k in {"BBB_Martins","Bioavailability_Ma","HIA_Hou"}:
            
                if k == "BBB_Martins" and not is_cns:
                    out[k] = _clamp01(1.0 - v)
                else:
                    out[k] = _clamp01(v)
                continue

            if k in _REG_RANGE:
                lo, hi = _REG_RANGE[k]
               
                out[k] = _scale_linear(v, lo, hi, invert=False)
                continue

       
        out[k] = 0.0
    return out

def multiobjective_vector_dat_admet(
    smiles_list,
    props=("dock", "qed", "sa", "aff", "logp"),
    *,
    pool,                      # DockingPool
    normalize: bool = True,
    return_raw: bool = True,
    logp_mu: float = 2.5,
    logp_sigma: float = 1.0,
    admet: bool = False,       # 开启则先跑 ADMET 过滤
    admet_project: str = "oral_peripheral",
    admet_model=None,
    admet_thr: float = 0.4,
):
   
    N = len(smiles_list)
    vectors, valid_smiles, is_valid = [], [], []
    raw_props = []

    toxicity_keys = ["hERG", "AMES", "DILI"]

    admet_rows = {}
    admet_ok = False
    if admet:
        try:
            if admet_model is None:
                from admet_ai import ADMETModel
                admet_model = ADMETModel()

            valid_for_admet, valid_indices = [], []
            for idx, s in enumerate(smiles_list):
                m = Chem.MolFromSmiles(s)
                if m is not None:
                    valid_for_admet.append(s)
                    valid_indices.append(idx)

            if valid_for_admet:
                admet_df = admet_model.predict(smiles=valid_for_admet)
                admet_rows = {i: admet_df.iloc[j].to_dict() for j, i in enumerate(valid_indices)}
                admet_ok = True
        except Exception as e:
            print(f"[WARN] ADMET prediction failed, skip pre-filter: {e}")
            admet_rows = {}
            admet_ok = False

    survivors_idx = []
    survivors_smiles = []
    filtered_idx = set()

    def _score_for_filter(row):
        scored = _score_admet_row(row, toxicity_keys, project=admet_project)
        for k in toxicity_keys:
            scored.setdefault(k, 0.0)
        return scored

    if admet and admet_ok:
        for i, s in enumerate(smiles_list):
            toxic_flag = False
            if i in admet_rows:
                scored = _score_for_filter(admet_rows[i])
                if (scored["hERG"] > admet_thr) or (scored["AMES"] > admet_thr) or (scored["DILI"] > admet_thr):
                    toxic_flag = True
            if toxic_flag:
                filtered_idx.add(i)
            else:
                survivors_idx.append(i)
                survivors_smiles.append(s)
    else:
        survivors_idx = list(range(N))
        survivors_smiles = list(smiles_list)

    results_by_idx = {}
    if len(survivors_smiles) > 0:
        dock_results = pool.score_batch(survivors_smiles)  
        for i_local, i_global in enumerate(survivors_idx):
            results_by_idx[i_global] = dock_results[i_local]

    for i, s in enumerate(smiles_list):
        admet_filtered = (i in filtered_idx)

        sample = {k: 0.0 for k in props}
        sample.update({"dock": 0.0, "qed": 0.0, "sa": 0.0, "aff": 0.0, "logp": 0.0})

        if admet_ok and (i in admet_rows):
            scored_all = _score_admet_row(admet_rows[i], [k for k in props if k not in {"dock","qed","sa","aff","logp"}] + toxicity_keys, project=admet_project)
            sample.update({k: float(scored_all.get(k, 0.0)) for k in scored_all})
            sample["_admet_raw"] = admet_rows[i]
            sample["_admet_filtered"] = bool(admet_filtered)

        if admet_filtered:
            vectors.append([0.0] * len(props))
            raw_props.append(sample)
            is_valid.append(False)
            valid_smiles.append(s)
            continue

        r = results_by_idx.get(i, None)
        if r is not None and r.get("ok", False):
            aff = float(r.get("aff", 0.0))
            qed = float(np.clip(r.get("qed", 0.0), 0.0, 1.0))
            sa_raw = r.get("sa", None)
            dock = _normalize_dock_affinity(aff)
            sa_n = float(np.clip(1.0 - (sa_raw / 10.0), 0.0, 1.0)) if sa_raw is not None else 0.0

            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                _logp = float(Descriptors.MolLogP(mol))
                score = np.exp(-0.5 * ((_logp - logp_mu) / max(1e-6, logp_sigma)) ** 2)
                logp = float(np.clip(score, 0.0, 1.0))
            else:
                logp = 0.0

            sample.update({"dock": dock, "qed": qed, "sa": sa_n, "aff": aff, "logp": logp})

            for k in props:
                sample.setdefault(k, 0.0)

            _update_running_stats({k: sample[k] for k in props})
            vec = [sample[k] for k in props]
            if normalize and all(_running_stats[k]["n"] > 10 for k in props):
                vec = _normalize_property_vector(vec, props)

            vectors.append(vec)
            raw_props.append(sample)
            is_valid.append(True)
            valid_smiles.append(s)
        else:
            vectors.append([0.0] * len(props))
            raw_props.append(sample)
            is_valid.append(False)
            valid_smiles.append(s)

    vectors = np.array(vectors, dtype=np.float32)
    if return_raw:
        return vectors, valid_smiles, is_valid, raw_props
    else:
        return vectors, valid_smiles, is_valid


def multiobjective_vector_dat_admet_and(
    smiles_list,
    props=("dock", "qed", "sa", "aff", "logp",
           # "hERG","AMES","DILI","CYP3A4_Veith","CYP2D6_Veith",
           # "HIA_Hou","Caco2_Wang","BBB_Martins"
    ),
    *,
    pool,  # DockingPool
    normalize: bool = True,
    return_raw: bool = True,
    logp_mu: float = 2.5,
    logp_sigma: float = 1.0,
    admet: bool = False,
    admet_project: str = "oral_peripheral",
    admet_model=None,         
    admet_thr = 0.4,
):
   
    results = pool.score_batch(smiles_list)

    admet_rows = None
    admet_keys = [k for k in props if k not in {"dock","qed","sa","aff","pKd_est","ki_est_nM","logp"}]
    if admet and len(admet_keys) > 0:
        try:
            if admet_model is None:
                from admet_ai import ADMETModel
                admet_model = ADMETModel()
            valid_for_admet = []
            valid_indices = []
            for idx, (s, r) in enumerate(zip(smiles_list, results)):
                if r is not None and r.get("ok", False):
                    mol = Chem.MolFromSmiles(s)
                    if mol is not None:
                        valid_for_admet.append(s)
                        valid_indices.append(idx)
            if valid_for_admet:
                admet_df = admet_model.predict(smiles=valid_for_admet)
                admet_rows = {i: admet_df.iloc[j].to_dict() for j, i in enumerate(valid_indices)}
            else:
                admet_rows = {}
        except Exception as e:
            admet_rows = {}
            print(f"[WARN] ADMET prediction failed: {e}")

    vectors, valid_smiles, is_valid = [], [], []
    raw_props = []

    for i, (s, r) in enumerate(zip(smiles_list, results)):
        if r is not None and r.get("ok", False):
            aff = float(r.get("aff", 0.0))
            qed = float(np.clip(r.get("qed", 0.0), 0.0, 1.0))
            sa_raw = r.get("sa", None)
            dock = _normalize_dock_affinity(aff)
            sa_n = float(np.clip(1.0 - (sa_raw / 10.0), 0.0, 1.0)) if sa_raw is not None else 0.0

            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                _logp = float(Descriptors.MolLogP(mol))
                score = np.exp(-0.5 * ((_logp - logp_mu) / max(1e-6, logp_sigma)) ** 2)
                logp = float(np.clip(score, 0.0, 1.0))
            else:
                logp = 0.0

            sample = {
                "dock": dock, "qed": qed, "sa": sa_n, "aff": aff, "logp": logp,
            }

            if admet and admet_rows is not None and i in admet_rows and len(admet_keys) > 0:
                admet_scored = _score_admet_row(admet_rows[i], admet_keys, project=admet_project)
                if admet_scored['hERG'] > admet_thr and admet_scored['AMES'] > admet_thr and admet_scored['DILI'] > admet_thr:
                    # for key in admet_scored.keys():
                    #     admet_scored[key] = 0
                    admet_scored = {k: 0 for k in admet_scored}
                    for key in sample:
                            sample[key] = 0
                sample.update(admet_scored)
                sample["_admet_raw"] = admet_rows[i]

            for k in props:
                sample.setdefault(k, 0.0)

            _update_running_stats({k: sample[k] for k in props})
            vec = [sample[k] for k in props]
            if normalize and all(_running_stats[k]["n"] > 10 for k in props):
                vec = _normalize_property_vector(vec, props)

            vectors.append(vec)
            raw_props.append(sample)
            is_valid.append(True)
            valid_smiles.append(s)
        else:
            sample = {k: 0.0 for k in props}
            sample.update({"dock":0.0,"qed":0.0,"sa":0.0,"aff":0.0,"logp":0.0})
            vectors.append([0.0]*len(props))
            raw_props.append(sample)
            is_valid.append(False)
            valid_smiles.append(s)

    vectors = np.array(vectors, dtype=np.float32)
    if return_raw:
        return vectors, valid_smiles, is_valid, raw_props
    else:
        return vectors, valid_smiles, is_valid

def scalarized_score(smiles: str, weights: dict):
    p = _calc_properties(smiles, props=list(weights.keys()))
    if p is None:
        return 0.0
    values = [max(1e-6, p.get(k, 0.0)) ** weights.get(k, 1.0) for k in weights]
    return float(np.prod(values) ** (1.0 / len(values)))



def get_running_stats():
    with _normalizer_lock:
        out = {}
        for k, st in _running_stats.items():
            var = st["M2"] / max(1, st["n"] - 1)
            out[k] = {"mean": st["mean"], "std": math.sqrt(var), "n": st["n"]}
        return out


def save_running_stats(path="stats_cache.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(get_running_stats(), f, indent=2)
    print(f"[scoring] Saved running stats → {path}")


def load_running_stats(path="stats_cache.json"):
    global _running_stats
    if not os.path.exists(path):
        print(f"[scoring] No stats cache found at {path}")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with _normalizer_lock:
        for k, v in data.items():
            _running_stats[k] = {
                "mean": v.get("mean", 0.0),
                "M2": (v.get("std", 1.0) ** 2) * max(1, v.get("n", 1) - 1),
                "n": v.get("n", 1),
            }
    print(f"[scoring] Loaded running stats from {path}")


def reset_running_stats():
    global _running_stats
    with _normalizer_lock:
        _running_stats = defaultdict(lambda: {"mean": 0.0, "M2": 0.0, "n": 0})
    print("[scoring] Running stats reset")



if __name__ == "__main__":
    base_path = '/home/wangqh/xk/FRATTVAE/runs/frattvae_pro_20251013_195244'
    json_path = os.path.join(base_path, 'topk.json')
    csv_output_path = os.path.join(base_path, 'topk_properties.csv')
    with open(json_path, 'r') as f:
        data = json.load(f)

    with open(csv_output_path, mode='w', newline='') as csv_file:
        fieldnames = ["smiles", "qed", "sa", "logp", "qeppi", "np", "qrci"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()

        for item in data:
            smiles = item[1]  
            properties = _calc_raw_properties(smiles)
            properties["smiles"] = smiles  
            writer.writerow(properties)

    print(f"Properties written to {csv_output_path}.")

    # print(len(data))
    # for item in data:
    #     value = item[0]
    #     smiles = item[1]
    #     out_dict = _calc_raw_properties(smiles)    #  props=("qed", "sa", "logp", "qeppi", "np", "qrci")


       
    