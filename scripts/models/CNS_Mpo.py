#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch CNS-MPO scorer (RDKit + ChemAxon cxcalc)
- Input: SDF or SMILES (.smi/.smiles; SMILES文件默认以 <SMILES><TAB><Name> 形式)
- Standardization: remove salts, keep largest fragment, remove stereochemistry (可选), uncharge (可选)
- Properties:
  * From RDKit: MW, TPSA, HBD, cLogP(Crippen)
  * From cxcalc: logD at pH (default 7.4), basic pKa (take the max basic site)
- Scoring: Wager CNS-MPO (0–6), 6 desirability functions summed
- Output: CSV with per-molecule details + histogram PNG + summary printed
"""

import argparse
import os
import sys
import tempfile
import subprocess
import shutil
import math

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, rdMolStandardize, rdMolOps

# ---------- CNS-MPO scoring functions (Wager et al.) ----------
def clip01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def score_ramp(x, a, b, reverse=False):
    """Linear desirability from 1 at 'a' to 0 at 'b'; clip to [0,1].
       If reverse=True, swap meaning (not used here)."""
    if reverse:
        a, b = b, a
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    return (b - x) / (b - a)

def score_tpsa(t):
    # 0 at <=20; linear up to 1 at 40; flat 40–90; linear down to 0 at 120; 0 beyond
    if t <= 20:
        return 0.0
    if t < 40:
        return (t - 20) / 20.0
    if t <= 90:
        return 1.0
    if t < 120:
        return (120.0 - t) / 30.0
    return 0.0

def cns_mpo_from_props(clogp, clogd74, mw, tpsa, hbd, pka_basic):
    s_clogp = clip01((5.0 - clogp) / 2.0)          # 1 at <=3; 0 at >=5
    s_clogd = clip01((4.0 - clogd74) / 2.0)        # 1 at <=2; 0 at >=4
    s_mw    = clip01((500.0 - mw) / 140.0)         # 1 at <=360; 0 at >=500
    s_hbd   = clip01((3.5 - hbd) / 3.0)            # 1 at <=0.5; 0 at >=3.5
    s_pka   = clip01((10.0 - pka_basic) / 2.0)     # 1 at <=8; 0 at >=10
    s_tpsa  = score_tpsa(tpsa)
    mpo = s_clogp + s_clogd + s_mw + s_tpsa + s_hbd + s_pka
    return mpo, dict(s_clogp=s_clogp, s_clogd=s_clogd, s_mw=s_mw,
                     s_tpsa=s_tpsa, s_hbd=s_hbd, s_pka=s_pka)

# ---------- cxcalc helpers ----------
def require_cxcalc():
    exe = shutil.which("cxcalc")
    if exe is None:
        sys.stderr.write("ERROR: 'cxcalc' not found in PATH. Please install ChemAxon Marvin.\n")
        sys.exit(2)
    return exe

def run_cxcalc_table(cmd_args, infile):
    """Run cxcalc and return TSV as pandas DataFrame (includes SMILES and name)."""
    proc = subprocess.run(["cxcalc", "-S", "-o", "tsv"] + cmd_args + [infile],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)
    if proc.returncode != 0:
        sys.stderr.write("cxcalc error:\n" + proc.stderr + "\n")
        raise RuntimeError("cxcalc failed: " + " ".join(cmd_args))
    # Robust read: allow ragged lines
    from io import StringIO
    df = pd.read_csv(StringIO(proc.stdout), sep="\t", engine="python")
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    # enforce presence
    if "name" not in df.columns:
        # Some cxcalc versions may output 'id' instead of 'name'
        if "id" in df.columns:
            df.rename(columns={"id": "name"}, inplace=True)
        else:
            raise ValueError("cxcalc output missing 'name' column.")
    return df

def parse_pka_basic(df_pka):
    """Row-wise max of all numeric 'basic pKa' columns."""
    numeric_cols = []
    for c in df_pka.columns:
        # keep columns that look like basic pKa values
        if "basic" in c and "pka" in c:
            numeric_cols.append(c)
    # Fallback: pick all numeric columns not named 'smiles'/'name'
    if not numeric_cols:
        numeric_cols = [c for c in df_pka.columns
                        if c not in ("smiles","name") and pd.api.types.is_numeric_dtype(df_pka[c])]
    if not numeric_cols:
        raise ValueError("Could not identify basic pKa columns in cxcalc output.")
    return df_pka[numeric_cols].max(axis=1, skipna=True)

# ---------- IO & standardization ----------
def load_molecules(path):
    ext = os.path.splitext(path)[1].lower()
    mols = []
    names = []
    if ext in [".sdf", ".sd"]:
        suppl = Chem.SDMolSupplier(path, removeHs=False)
        for i, m in enumerate(suppl):
            if m is None:
                continue
            name = m.GetProp("_Name") if m.HasProp("_Name") else f"mol_{i+1}"
            mols.append(m); names.append(name)
    elif ext in [".smi", ".smiles", ".txt", ".tsv", ".csv"]:
        # Expect SMILES \t NAME; tolerate whitespace
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p for p in line.replace(",", "\t").split("\t") if p!=""]
                smi = parts[0]
                name = parts[1] if len(parts) > 1 else f"mol_{i}"
                m = Chem.MolFromSmiles(smi)
                if m is None:
                    continue
                m.SetProp("_Name", name)
                mols.append(m); names.append(name)
    else:
        raise ValueError("Unsupported input type: {}".format(ext))
    return mols, names

def standardize_mol(mol, remove_salts=True, remove_stereo=True, uncharge=False):
    m = Chem.Mol(mol)
    # keep largest fragment & remove salts
    if remove_salts:
        lfc = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
        m = lfc.choose(m)
        remover = rdMolStandardize.SaltRemover()  # default salt list
        m = remover.StripMol(m, dontRemoveEverything=True)
    if uncharge:
        uncharger = rdMolStandardize.Uncharger()
        m = uncharger.uncharge(m)
    if remove_stereo:
        rdMolOps.RemoveStereochemistry(m)
    Chem.SanitizeMol(m)
    return m

def write_smiles_tmp(mols, names, path):
    with open(path, "w", encoding="utf-8") as f:
        for m, n in zip(mols, names):
            smi = Chem.MolToSmiles(m, isomericSmiles=False)  # no stereo
            f.write(f"{smi}\t{n}\n")

# ---------- Main pipeline ----------
def main():
    ap = argparse.ArgumentParser(description="Compute CNS-MPO (Wager) for a molecule library.")
    ap.add_argument("input", help="Input file (.sdf / .smi)")
    ap.add_argument("--ph", type=float, default=7.4, help="pH for logD (default: 7.4)")
    ap.add_argument("--no-remove-salts", action="store_true", help="Do NOT remove salts / keep multi-fragment")
    ap.add_argument("--keep-stereo", action="store_true", help="Do NOT remove stereochemistry")
    ap.add_argument("--uncharge", action="store_true", help="Neutralize charges (RDKit Uncharger)")
    ap.add_argument("--out-csv", default="cns_mpo_results.csv", help="Output CSV file")
    ap.add_argument("--hist-png", default="cns_mpo_hist.png", help="Histogram PNG file")
    ap.add_argument("--bins", type=int, default=30, help="Histogram bins (default 30)")
    args = ap.parse_args()

    require_cxcalc()

    mols, names = load_molecules(args.input)
    if not mols:
        print("No valid molecules parsed.", file=sys.stderr); sys.exit(1)
    # Standardize
    std_mols = [standardize_mol(m,
                                remove_salts=not args.no_remove_salts,
                                remove_stereo=not args.keep_stereo,
                                uncharge=args.uncharge) for m in mols]

    # Prepare temporary SMILES for cxcalc
    with tempfile.TemporaryDirectory() as td:
        smi_path = os.path.join(td, "input.smi")
        write_smiles_tmp(std_mols, names, smi_path)

        # Run cxcalc for logP / logD / pKa (basic)
        df_logp = run_cxcalc_table(["logp"], smi_path)
        df_logd = run_cxcalc_table(["logd", "-H", str(args.ph)], smi_path)
        df_pka  = run_cxcalc_table(["pka", "-b"], smi_path)

    # Build a combined frame keyed by 'name'
    def pick_numeric_last(df, prefer_contains=None):
        # get the rightmost numeric column (e.g., 'logd(7.4)' etc.)
        candidates = [c for c in df.columns if c not in ("smiles","name") and pd.api.types.is_numeric_dtype(df[c])]
        if prefer_contains:
            pref = [c for c in candidates if prefer_contains in c]
            if pref:
                candidates = pref
        if not candidates:
            raise ValueError("No numeric property columns found in cxcalc output.")
        return candidates[-1]

    col_logp = pick_numeric_last(df_logp, prefer_contains="logp")
    col_logd = pick_numeric_last(df_logd)  # likely 'logd(' in name, but be general
    pkabasic_series = parse_pka_basic(df_pka)

    df_cx = pd.DataFrame({
        "name": df_logp["name"],
        "cx_logp": pd.to_numeric(df_logp[col_logp], errors="coerce"),
        "cx_logd_ph": pd.to_numeric(df_logd[col_logd], errors="coerce"),
        "pka_basic": pd.to_numeric(pkabasic_series, errors="coerce"),
    })

    # RDKit descriptors
    rows = []
    for m, n in zip(std_mols, names):
        try:
            mw = Descriptors.MolWt(m)
            tpsa = rdMolDescriptors.CalcTPSA(m)
            hbd = Lipinski.NHOHDonors(m)
            # RDKit's Crippen logP; keep as independent column for参考
            clogp_rdkit = Descriptors.MolLogP(m)
            rows.append({"name": n, "mw": mw, "tpsa": tpsa, "hbd": float(hbd), "clogp_rdkit": clogp_rdkit})
        except Exception as e:
            rows.append({"name": n, "mw": None, "tpsa": None, "hbd": None, "clogp_rdkit": None})
    df_rd = pd.DataFrame(rows)

    # Merge
    df = df_rd.merge(df_cx, on="name", how="left")

    # Choose which cLogP to use in MPO: use ChemAxon logP for consistency with cxcalc.
    df["clogp"] = df["cx_logp"]
    df["clogd74"] = df["cx_logd_ph"]

    # Score
    scores = []
    for i, r in df.iterrows():
        try:
            mpo, parts = cns_mpo_from_props(
                clogp=float(r["clogp"]),
                clogd74=float(r["clogd74"]),
                mw=float(r["mw"]),
                tpsa=float(r["tpsa"]),
                hbd=float(r["hbd"]),
                pka_basic=float(r["pka_basic"])
            )
        except Exception:
            mpo, parts = (float("nan"), {k: float("nan") for k in ["s_clogp","s_clogd","s_mw","s_tpsa","s_hbd","s_pka"]})
        scores.append({"cns_mpo": mpo, **parts})
    df_scores = pd.DataFrame(scores)
    df_out = pd.concat([df, df_scores], axis=1)

    # Summary
    valid = df_out["cns_mpo"].dropna()
    mean_val = valid.mean() if len(valid) else float("nan")
    median_val = valid.median() if len(valid) else float("nan")

    # Save CSV
    df_out_cols = [
        "name","clogp","clogd74","pka_basic","mw","tpsa","hbd",
        "cns_mpo","s_clogp","s_clogd","s_mw","s_tpsa","s_hbd","s_pka",
        "clogp_rdkit","cx_logp","cx_logd_ph"
    ]
    df_out[df_out_cols].to_csv(args.out_csv, index=False)
    print(f"[✔] Wrote per-molecule results: {args.out_csv}")
    print(f"[✔] CNS-MPO mean = {mean_val:.4f}, median = {median_val:.4f} (N={len(valid)})")

    # Histogram
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt

        plt.figure()
        plt.hist(valid, bins=args.bins)
        plt.xlabel("CNS-MPO score (0–6)")
        plt.ylabel("Count")
        plt.title("CNS-MPO Distribution")
        plt.tight_layout()
        plt.savefig(args.hist_png, dpi=150)
        print(f"[✔] Wrote histogram: {args.hist_png}")
    except Exception as e:
        print(f"[!] Could not write histogram: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
