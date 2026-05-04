import os, json, argparse
import pandas as pd
from admet_ai import ADMETModel

def _clamp01(x):
    try:
        return float(min(1.0, max(0.0, x)))
    except Exception:
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
    "Solubility_AqSolDB": (-12.0, 0.0),      
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
    """
    row_dict: ADMET-AI 返回的一行（字典）
    keys: 需要输出的端点名
    project: 'oral_cns' 表示中枢；否则视作外周
    """
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

        if k in row_dict:
            v = row_dict[k]
            if isinstance(v, (int, float)):
                v = float(v)
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

def main():
    parser = argparse.ArgumentParser(description='Run ADMET-AI and append selected endpoints to CSV')
    parser.add_argument('--base_path', type=str, required=True,
                        help='Folder containing the input CSV')
    parser.add_argument('--csv_name', type=str, required=True,
                        help='Input CSV filename')
    parser.add_argument('--admet_keys', type=str, required=True,
                        help='Path to a JSON file: list or dict of endpoint names to add')
    parser.add_argument('--smiles_col', type=str, default='SMILES',
                        help='Name of the SMILES column (if not provided, auto-detect)')
    parser.add_argument('--project', type=str, default='oral_pg',
                        help="Project context: 'oral_cns' or 'oral_pg' etc., affects BBB/Pgp direction")
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV filename (default: <input>_admet.csv)')
    args = parser.parse_args()

    in_csv = os.path.join(args.base_path, args.csv_name)
    if not os.path.isfile(in_csv):
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    with open(args.admet_keys, 'r') as f:
        ak = json.load(f)
    if isinstance(ak, dict):
        admet_keys = list(ak.keys())
    elif isinstance(ak, list):
        admet_keys = list(ak)
    else:
        raise ValueError("admet_keys JSON must be a list or dict.")

    df = pd.read_csv(in_csv)
    if args.smiles_col:
        smi_col = args.smiles_col
        if smi_col not in df.columns:
            raise ValueError(f"Specified smiles_col '{smi_col}' not in columns: {list(df.columns)}")
    else:
        candidates = ["smiles","SMILES","Smiles","canonical_smiles"]
        found = [c for c in candidates if c in df.columns]
        if not found:
            raise ValueError(f"No SMILES column found. Tried: {candidates}")
        smi_col = found[0]

    smiles_series = df[smi_col].astype(str)
    mask_valid = smiles_series.notna() & (smiles_series.str.len() > 0)
    idx_valid = df.index[mask_valid].tolist()
    smiles_valid = smiles_series.loc[idx_valid].tolist()
    if len(smiles_valid) == 0:
        raise ValueError("No valid (non-empty) SMILES found to run ADMET.")

    model = ADMETModel()
    admet_df = model.predict(smiles=smiles_valid)   

  
    for k in admet_keys:
        if k not in df.columns:
            df[k] = pd.NA

    for j, idx in enumerate(idx_valid):
        row_raw = admet_df.iloc[j].to_dict()
        scored = _score_admet_row(row_raw, admet_keys, project=args.project)
        for k, v in scored.items():
            df.at[idx, k] = float(v).__format__('.4f')

    if args.output:
        out_csv = args.output
    else:
        stem, ext = os.path.splitext(args.csv_name)
        out_csv = os.path.join(args.base_path, f"{stem}_admet{ext}")

    df.to_csv(out_csv,  float_format='%.4f', index=False)
    print(f"[OK] ADMET columns added: {admet_keys}")
    print(f"[OK] Saved to: {out_csv}")

if __name__ == "__main__":
    main()


# python add_admet_to_csv.py \
#   --base_path /raid/home/xukai/FRATTVAE/results/CNS_SMILES_standardized_struct_1020/generate \
#   --csv_name generate_frattvae_pro_lrrk2_20251025_104921_policy_250_20251026_211333_1.csv \
#   --admet_keys /raid/home/xukai/FRATTVAE/results/CNS_SMILES_standardized_struct_1020/input_data/admet_par.json \
#   --project oral_cns
