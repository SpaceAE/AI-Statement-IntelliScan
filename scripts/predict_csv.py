#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pandas as pd
import joblib

REQ_COLS = [
    "debit_amount","credit_amount","balance_amount","net_amount","abs_amount",
    "log1p_amount","hour","dayofweek","is_weekend","day","month","year",
    "tx_code","channel","description_text"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_bundle", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    bundle = joblib.load(args.model_bundle)
    pipe = bundle["pipeline"]
    thr  = float(bundle.get("threshold", 0.5))

    df = pd.read_csv(args.input_csv)
    # เตรียมคอลัมน์ขาด/ชนิดข้อมูลขั้นต่ำ (กันพลาดเบื้องต้น)
    for c in REQ_COLS:
        if c not in df.columns:
            df[c] = "" if c in ("tx_code","channel","description_text") else 0
    # แปลงบางคอลัมน์เป็นตัวเลข ถ้าจำเป็น
    for c in ["debit_amount","credit_amount","balance_amount","net_amount","abs_amount","log1p_amount",
              "hour","dayofweek","is_weekend","day","month","year"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # ทำนาย
    if hasattr(pipe, "predict_proba"):
        p = pipe.predict_proba(df)[:,1]
    else:
        s = pipe.decision_function(df)
        p = (s - s.min()) / (s.max() - s.min() + 1e-9)

    yhat = (p >= thr).astype(int)

    out = df.copy()
    out["risk_score"] = p
    out["risk_label"] = yhat
    out.to_csv(args.output_csv, index=False)
    print("[WROTE]", args.output_csv)

if __name__ == "__main__":
    main()
