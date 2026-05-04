#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse
import random
import pandas as pd
import numpy as np

# ---------------------- 预置配置（静态方案） ----------------------
PROFILES = {
    "broad": {   # A. 通用-宽松
        "natoms":  (12, 46),
        "MW":      (150, 650),
        "logP":    (0.0, 5.5),
        "QED":     (0.35, 1.0),
        "SA":      (0.0, 6.0),
        "NP":      (-3.0, 1.1),
        "TPSA":    (10.0, 160.0),
        "BertzCT": (250.0, 1800.0),
    },
    "oral": {    # B. 口服-药样
        "natoms":  (12, 46),
        "MW":      (150, 500),
        "logP":    (0.0, 5.0),
        "QED":     (0.40, 1.0),
        "SA":      (0.0, 5.5),
        "NP":      (-2.5, 1.0),
        "TPSA":    (20.0, 140.0),
        "BertzCT": (250.0, 1700.0),
    },
    "cns": {     # C. 口服-中枢
        "natoms":  (16, 42),
        "MW":      (200, 450),
        "logP":    (1.5, 4.5),
        "QED":     (0.45, 1.0),
        "SA":      (0.0, 5.5),
        "NP":      (-2.5, 1.0),
        "TPSA":    (10.0, 90.0),
        "BertzCT": (250.0, 1700.0),
    },
    "lipinski": {  # 经典 RO5（不使用 HBD/HBA 作为过滤结果中的参与？→ 这里参与；你也可用作第二阶段）
        "MW":   (None, 500.0),   # ≤ 500
        "logP": (None, 5.0),     # ≤ 5
        "HBD":  (None, 5.0),     # ≤ 5
        "HBA":  (None, 10.0),    # ≤ 10
    }
}

NUMERIC_COLS = ["natoms","MW","logP","QED","SA","NP","TPSA","BertzCT"]  # 动态阈值默认覆盖的列
SMILES_CANDIDATES = ["SMILES", "smiles", "Smiles", "canonical_smiles"]

# ---------------------- RDKit: HBD/HBA 计算 ----------------------
def compute_hbd_hba(df, smiles_col):
    try:
        from rdkit import Chem
        from rdkit.Chem import Lipinski
    except Exception as e:
        print(f"[WARN] RDKit not available, HBD/HBA will be NaN: {e}")
        df["HBD"] = pd.NA
        df["HBA"] = pd.NA
        return df

    def _one(s):
        try:
            m = Chem.MolFromSmiles(str(s))
            if m is None: return (np.nan, np.nan)
            return (float(Lipinski.NumHDonors(m)), float(Lipinski.NumHAcceptors(m)))
        except Exception:
            return (np.nan, np.nan)

    vals = [ _one(s) for s in df[smiles_col].astype(str) ]
    df["HBD"] = [v[0] for v in vals]
    df["HBA"] = [v[1] for v in vals]
    return df

# ---------------------- 动态阈值生成（mean ± kσ） ----------------------
def dynamic_ranges(df, cols, k=2.0, floor=None, ceil=None):
    """
    返回 {col: (lo, hi)}，lo/hi = mean ± k*std，并截断到数据 min/max 或给定 floor/ceil。
    """
    ranges = {}
    for col in cols:
        if col not in df.columns: 
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty: 
            continue
        mu, sd = float(s.mean()), float(s.std(ddof=0))
        lo, hi = mu - k*sd, mu + k*sd
        # 限定到观测范围
        lo = max(lo, float(s.min()))
        hi = min(hi, float(s.max()))
        # 如果提供了绝对上下限，再裁一次
        if floor and col in floor and floor[col] is not None:
            lo = max(lo, floor[col])
        if ceil and col in ceil and ceil[col] is not None:
            hi = min(hi, ceil[col])
        ranges[col] = (lo, hi)
    return ranges

# ---------------------- 过滤工具 ----------------------
def mask_by_ranges(df, ranges: dict):
    parts = []
    for col, (lo, hi) in ranges.items():
        if col not in df.columns: 
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        # None 表示单边限制
        m = pd.Series(True, index=df.index)
        if lo is not None:
            m &= (s >= lo)
        if hi is not None:
            m &= (s <= hi)
        parts.append(m)
    return np.logical_and.reduce(parts) if parts else pd.Series(True, index=df.index)

def profile_to_ranges(df, profile_name: str):
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}")
    return PROFILES[profile_name]

# ---------------------- 主流程 ----------------------
def main():
    ap = argparse.ArgumentParser(description="两阶段过滤：Stage1(可动态 mean±kσ) -> Stage2(静态档位)，并写入HBD/HBA")
    ap.add_argument("--in_csv", required=True, help="输入 CSV，需含列：ID,SMILES,test,natoms,MW,logP,QED,SA,NP,TPSA,BertzCT")
    ap.add_argument("--out_csv", default=None, help="输出 CSV（默认在名称后附加后缀）")
    ap.add_argument("--smiles_col", default=None, help="SMILES 列名，默认自动猜测")

    # 阶段控制
    ap.add_argument("--stage1_profile", choices=["broad","oral","cns","lipinski","none"], default="broad",
                    help="第一阶段静态方案；当 --stage1_dynamic 为 True 时将被动态阈值覆盖")
    ap.add_argument("--stage1_dynamic", action="store_true", help="启用第一阶段 mean±kσ 动态阈值")
    ap.add_argument("--sigma_k", type=float, default=2.0, help="动态阈值的 k 值，默认 2.0")

    ap.add_argument("--stage2_profile", choices=["none","oral","cns","lipinski","broad"], default="none",
                    help="第二阶段静态方案；none 表示不进行第二阶段过滤")

    args = ap.parse_args()

    # 读入
    df = pd.read_csv(args.in_csv)
    df = df.drop(columns=["test"])
    # SMILES 列
    if args.smiles_col:
        smi_col = args.smiles_col
        if smi_col not in df.columns:
            raise ValueError(f"指定的 SMILES 列不存在：{smi_col}")
    else:
        cands = [c for c in SMILES_CANDIDATES if c in df.columns]
        if not cands:
            raise ValueError(f"未找到 SMILES 列，尝试了：{SMILES_CANDIDATES}")
        smi_col = cands[0]

    # 计算 HBD/HBA（不参与过滤）
    df = compute_hbd_hba(df, smi_col)

    # ---------- Stage 1 ----------
    if args.stage1_dynamic:
        # 动态范围：对 NUMERIC_COLS 可用列计算 mean±kσ
        avail_cols = [c for c in NUMERIC_COLS if c in df.columns]
        dyn = dynamic_ranges(df, avail_cols, k=args.sigma_k)
        m1 = mask_by_ranges(df, dyn)
        stage1_info = f"Stage1 dynamic mean±{args.sigma_k}σ on {list(dyn.keys())}"
    elif args.stage1_profile != "none":
        r1 = profile_to_ranges(df, args.stage1_profile)
        m1 = mask_by_ranges(df, r1)
        stage1_info = f"Stage1 profile={args.stage1_profile}"
    else:
        m1 = pd.Series(True, index=df.index)
        stage1_info = "Stage1 skipped"

    df1 = df[m1].copy()

    # ---------- Stage 2 ----------
    if args.stage2_profile != "none":
        r2 = profile_to_ranges(df1, args.stage2_profile)
        m2 = mask_by_ranges(df1, r2)
        df2 = df1[m2].copy()
        stage2_info = f"Stage2 profile={args.stage2_profile}"
    else:
        df2 = df1
        stage2_info = "Stage2 skipped"

    # 四舍五入（只对数值列），保存 3 位小数
    for col in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[col]):
            df2[col] = pd.to_numeric(df2[col], errors="coerce").round(3)
    
    random.seed(0)
    test = random.choices([0, 1, -1], k= len(df2), weights= [0.90, 0.05, 0.05])
    print('random train-valid-test split. train:valid:test= 0.90:0.05:0.05', flush= True)

    # add columns
    df2['test'] = test
    # 输出
    if not args.out_csv:
        stem, ext = os.path.splitext(args.in_csv)
        suffix = []
        suffix.append("dyn" if args.stage1_dynamic else args.stage1_profile)
        suffix.append(args.stage2_profile)
        out_csv = f"{stem}_f_{'_'.join(suffix)}{ext}"
    else:
        out_csv = args.out_csv

    df2.to_csv(out_csv, index=False, float_format="%.3f")

    # 报告
    print(f"[OK] {stage1_info}  ->  kept {len(df1)}/{len(df)} ({len(df1)/max(1,len(df)):.1%})")
    print(f"[OK] {stage2_info}  ->  kept {len(df2)}/{len(df1)} ({len(df2)/max(1,len(df1)):.1%})")
    print(f"[OK] HBD/HBA added (not used for filtering). Saved to: {out_csv}")

if __name__ == "__main__":
    main()


# python filter_chembl_36.py \
#   --in_csv chembl_36_20251023_r1r10w1100w900_standardized.csv \
#   --stage1_dynamic --sigma_k 2.0 \
#   --stage2_profile oral


# python filter_chembl_36.py \
#   --in_csv chembl_36_20251023_r1r10w1100w900_standardized.csv \
#   --stage1_profile broad \
#   --stage2_profile cns


# python filter_chembl_36.py \
#   --in_csv chembl_36_20251023_r1r10w1100w900_standardized.csv \
#   --stage1_profile lipinski \
#   --stage2_profile none
