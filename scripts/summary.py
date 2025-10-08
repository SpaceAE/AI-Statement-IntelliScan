#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
from pathlib import Path

EXP_DIR = Path("experiments")
RESULT_CSV = EXP_DIR / "exp_results.csv"
GRID_CROSS = EXP_DIR / "grid_cross_model.csv"

PRINT_TOP_K = 20
TARGET_NSPLITS = 5  # อยากกรองเฉพาะ n_splits=5

def safe_float(s, default=np.nan):
    try:
        return float(s)
    except Exception:
        return default

def ensure_columns(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def parse_params_json(df: pd.DataFrame) -> pd.DataFrame:
    if "params_json" not in df.columns:
        return df
    def parse_one(s):
        try:
            return json.loads(s)
        except Exception:
            return {}
    params = df["params_json"].map(parse_one).apply(pd.Series)
    # prefix เพื่อกันชนชื่อคอลัมน์
    params = params.add_prefix("p_")
    return pd.concat([df.drop(columns=["params_json"]), params], axis=1)

def main():
    if not RESULT_CSV.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ผลลัพธ์: {RESULT_CSV}")

    df = pd.read_csv(RESULT_CSV)

    # ทำความสะอาดคอลัมน์ที่เราจะใช้
    must_have = [
        "model",
        "cv_mean_pr_auc","cv_std_pr_auc",
        "cv_mean_mcc","cv_std_mcc",
        "cv_mean_bal_accuracy","cv_std_bal_accuracy",
        "cv_mean_accuracy","cv_std_accuracy",
        "inference_time_ms_per_1k","train_time_sec",
        "tfidf_ngram_range","tfidf_max_features","tfidf_min_df","tfidf_max_df",
    ]
    df = ensure_columns(df, must_have)

    # ถ้าไม่มี n_splits ให้เดาจาก grid_cross_model.csv (กรณี grid มีค่าเดียวทั้งไฟล์)
    if "n_splits" not in df.columns:
        inferred = None
        if GRID_CROSS.exists():
            g = pd.read_csv(GRID_CROSS)
            if "n_splits" in g.columns and g["n_splits"].nunique() == 1:
                inferred = int(g["n_splits"].iloc[0])
        if inferred is not None:
            df["n_splits"] = inferred
            print(f"[INFO] ไม่มีคอลัมน์ n_splits ในผลลัพธ์ → ใส่ค่าอนุมานจาก grid_cross_model.csv = {inferred}")
        else:
            print("[WARN] exp_results.csv ไม่มีคอลัมน์ n_splits และเดาไม่ได้จากกริด → จะไม่กรองด้วย n_splits")

    # กรองให้แฟร์ (ถ้าและเฉพาะเมื่อมีคอลัมน์ n_splits)
    if "n_splits" in df.columns:
        before = len(df)
        df = df[df["n_splits"].astype("float").fillna(-1) == TARGET_NSPLITS]
        print(f"[INFO] Filter by n_splits={TARGET_NSPLITS}: {before} → {len(df)} แถว")

    # ตัดแถวที่ไม่มี PR-AUC
    df = df[df["cv_mean_pr_auc"].notna()].copy()
    if len(df) == 0:
        raise SystemExit("[STOP] ไม่มีแถวที่มี cv_mean_pr_auc ให้จัดอันดับ")

    # บังคับชนิดตัวเลขที่ใช้ rank
    for c in [
        "cv_mean_pr_auc","cv_std_pr_auc",
        "cv_mean_mcc","cv_std_mcc",
        "cv_mean_bal_accuracy","cv_std_bal_accuracy",
        "cv_mean_accuracy","cv_std_accuracy",
        "inference_time_ms_per_1k","train_time_sec",
    ]:
        df[c] = df[c].apply(safe_float)

    # แตก params_json เป็นคอลัมน์ย่อย
    df = parse_params_json(df)

    # ===== Ranking (Borda-style) =====
    # ยิ่ง rank ต่ำ = ยิ่งดี
    df["r_pr"]   = df["cv_mean_pr_auc"].rank(ascending=False, method="min")
    df["r_mcc"]  = df["cv_mean_mcc"].rank(ascending=False, method="min")
    df["r_bal"]  = df["cv_mean_bal_accuracy"].rank(ascending=False, method="min")
    df["r_std1"] = df["cv_std_pr_auc"].rank(ascending=True,  method="min")  # std ต่ำดี
    df["r_std2"] = df["cv_std_mcc"].rank(ascending=True,     method="min")

    # น้ำหนักเริ่มต้น (ปรับได้ตามนโยบาย)
    df["rank_total"] = (
        2.0*df["r_pr"] +
        1.5*df["r_mcc"] +
        1.0*df["r_bal"] +
        0.5*df["r_std1"] +
        0.5*df["r_std2"]
    )

    # ===== แสดงผล =====
    show_cols = [
        "rank_total","model",
        "tfidf_ngram_range","tfidf_max_features","tfidf_min_df",
        "cv_mean_pr_auc","cv_std_pr_auc",
        "cv_mean_mcc","cv_std_mcc",
        "cv_mean_bal_accuracy","cv_std_bal_accuracy",
        "cv_mean_accuracy","cv_std_accuracy",
        "inference_time_ms_per_1k","train_time_sec",
        "run_id"
    ]
    show_cols = [c for c in show_cols if c in df.columns]

    print("\n=== TOP-ALL ===")
    print(df.sort_values("rank_total").head(PRINT_TOP_K)[show_cols].to_string(index=False))

    # Best by model
    if "model" in df.columns:
        idx = df.groupby("model")["rank_total"].idxmin()
        best_by_model = df.loc[idx].sort_values("rank_total")
        print("\n=== BEST BY MODEL ===")
        print(best_by_model[show_cols].to_string(index=False))

    # Best by TF-IDF combo
    tfidf_key = (
        df["tfidf_ngram_range"].astype(str)
        + "|maxF=" + df["tfidf_max_features"].astype(str)
        + "|min_df=" + df["tfidf_min_df"].astype(str)
    )
    df = df.assign(tfidf_key=tfidf_key)
    idx2 = df.groupby("tfidf_key")["rank_total"].idxmin()
    best_by_tfidf = df.loc[idx2].sort_values("rank_total")
    print("\n=== BEST BY TF-IDF COMBO ===")
    tfidf_cols = ["rank_total","tfidf_key","model","cv_mean_pr_auc","cv_mean_mcc","cv_mean_bal_accuracy"]
    tfidf_cols = [c for c in tfidf_cols if c in best_by_tfidf.columns]
    print(best_by_tfidf[tfidf_cols].to_string(index=False))

    # Hint: ถ้าอยาก export ตารางต่างๆ เป็น CSV
    # best_by_model.to_csv(EXP_DIR / "summary_best_by_model.csv", index=False)
    # best_by_tfidf.to_csv(EXP_DIR / "summary_best_by_tfidf.csv", index=False)

if __name__ == "__main__":
    main()
