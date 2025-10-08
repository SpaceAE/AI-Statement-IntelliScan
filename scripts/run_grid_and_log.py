"""
Run grid experiments and append results into a log CSV (exp_results.csv).
- Cross-model configs are read from grid_cross_model.csv
- Per-model grids are read from grid_logreg.csv, grid_randomforest.csv, grid_mlp.csv
- Results are appended to exp_results.csv (PR-AUC as primary, plus ACC/BalACC/MCC)
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import FitFailedWarning

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, make_scorer

# Always show which folds/configs failed (useful when error_score=np.nan)
warnings.filterwarnings("always", category=FitFailedWarning)

# ------------------------- Data Loading & Preprocess -------------------------

def load_all_statements(data_dir: str) -> pd.DataFrame:
    import glob
    import os
    paths = sorted(
        glob.glob(os.path.join(data_dir, "*.csv"))
        + glob.glob(os.path.join(data_dir, "*.xlsx"))
    )
    if not paths:
        raise FileNotFoundError(f"ไม่พบไฟล์ใน {data_dir}")
    dfs = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        df = pd.read_excel(p, engine="openpyxl") if ext == ".xlsx" else pd.read_csv(p)
        df["file_id"] = os.path.basename(p)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # datetime
    df["tx_datetime"] = pd.to_datetime(df["tx_datetime"], errors="coerce")
    df = (
        df.dropna(subset=["tx_datetime"])
        .sort_values(["file_id", "tx_datetime"])
        .reset_index(drop=True)
    )
    # split code/channel
    sp = df["code_channel_raw"].astype(str).str.split("/", n=1, expand=True)
    df["tx_code"] = sp[0].str.strip()
    df["channel"] = sp[1].str.strip() if sp.shape[1] > 1 else ""

    # numeric ensure + cast
    for col in ["debit_amount", "credit_amount", "balance_amount"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # engineered
    df["net_amount"] = df["credit_amount"] - df["debit_amount"]
    df["abs_amount"] = df["debit_amount"].abs() + df["credit_amount"].abs()
    df["log1p_amount"] = np.log1p(df["abs_amount"])

    # time features
    dt = df["tx_datetime"]
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year

    # text & label
    df["description_text"] = df["description_text"].astype(str).fillna("")
    df["fraud_label"] = df["fraud_label"].astype(int)
    return df


# ------------------------- Config → Preprocessor / Model -------------------------

NUMERIC_FEATURES = [
    "debit_amount", "credit_amount", "balance_amount",
    "net_amount", "abs_amount", "log1p_amount",
    "hour", "dayofweek", "is_weekend", "day", "month", "year",
]
CATEGORICAL_FEATURES = ["tx_code", "channel"]
TEXT_FEATURE = "description_text"


def build_preprocessor(tfidf_cfg: dict) -> ColumnTransformer:
    # TF-IDF config
    ngram = tfidf_cfg.get("tfidf_ngram_range", "1-2")
    if isinstance(ngram, str) and "-" in ngram:
        a, b = ngram.split("-")
        ngram_range = (int(a), int(b))
    else:
        ngram_range = (1, 2)

    max_features = int(float(tfidf_cfg.get("tfidf_max_features", 5000)))
    min_df = int(float(tfidf_cfg.get("tfidf_min_df", 5)))
    max_df = float(tfidf_cfg.get("tfidf_max_df", 0.95))

    text_transformer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        strip_accents="unicode",
    )

    # OneHot encoder
    try:
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

    numeric_transformer = StandardScaler(with_mean=True, with_std=True)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", cat_encoder, CATEGORICAL_FEATURES),
            ("txt", text_transformer, TEXT_FEATURE),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return preprocessor


def _parse_none_int_float(value, allow_none=True):
    """Helper: parse value that may be '', 'None', 'nan', int-like or float-like string."""
    if value is None:
        return None if allow_none else value
    s = str(value).strip()
    if allow_none and s in {"", "None", "nan", "NaN"}:
        return None
    # try integer then float
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return value


def build_model(model_key: str, model_params: dict) -> Any:
    key = model_key.upper()
    if key in ("LR", "LOGREG", "LOGISTIC", "LOGISTICREGRESSION"):
        clf = LogisticRegression(
            solver=model_params.get("solver", "saga"),
            penalty=model_params.get("penalty", "l2"),
            C=float(model_params.get("C", 1.0)),
            max_iter=int(model_params.get("max_iter", 500)),
            n_jobs=-1,
            class_weight=model_params.get("class_weight", "balanced"),
            random_state=42,
        )
    elif key in ("RF", "RANDOMFOREST", "RANDOM_FOREST"):
        # max_depth
        max_depth_val = _parse_none_int_float(model_params.get("max_depth", None), allow_none=True)
        if max_depth_val is not None:
            max_depth_val = int(float(max_depth_val))

        # max_features: float in (0,1], int >=1, 'sqrt'/'log2', or None
        mf_raw = model_params.get("max_features", "sqrt")
        if mf_raw is None:
            mf_val = None
        else:
            s = str(mf_raw).strip()
            if s in {"sqrt", "log2"}:
                mf_val = s
            elif s in {"", "None", "nan", "NaN"}:
                mf_val = None
            else:
                # "0.7" -> 0.7, "10" -> 10
                mf_val = _parse_none_int_float(s, allow_none=True)
                if isinstance(mf_val, str):
                    # fallback: try float
                    mf_val = float(mf_val)

        clf = RandomForestClassifier(
            n_estimators=int(float(model_params.get("n_estimators", 500))),
            max_depth=max_depth_val,
            min_samples_leaf=int(float(model_params.get("min_samples_leaf", 1))),
            max_features=mf_val,
            class_weight=model_params.get("class_weight", "balanced"),
            n_jobs=-1,
            random_state=42,
        )
    elif key in ("MLP",):
        hls = model_params.get("hidden_layer_sizes", "128")
        if isinstance(hls, str):
            sizes = tuple(int(x.strip()) for x in hls.split(",") if x.strip())
        else:
            sizes = tuple(hls)
        clf = MLPClassifier(
            hidden_layer_sizes=sizes if sizes else (128,),
            activation=model_params.get("activation", "relu"),
            alpha=float(model_params.get("alpha", 1e-4)),
            max_iter=int(float(model_params.get("max_iter", 200))),
            early_stopping=bool(model_params.get("early_stopping", True)),
            validation_fraction=float(model_params.get("validation_fraction", 0.1)),
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model key: {model_key}")
    return clf


# ------------------------- CV & Logging -------------------------

def run_cv_and_log(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    preprocessor: ColumnTransformer,
    model_key: str,
    model_params: dict,
    n_splits: int,
    exp_log_path: Path,
    cross_cfg: dict,
    run_id: str,
    random_state: int = 42,
) -> dict:
    pipe = Pipeline([("prep", preprocessor), ("clf", build_model(model_key, model_params))])

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "mcc": make_scorer(matthews_corrcoef),
        "ap": "average_precision",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
        "roc_auc": "roc_auc",
    }

    cv = GroupKFold(n_splits=n_splits)
    t0 = time.time()
    cvres = cross_validate(
        pipe,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
        verbose=0,
        error_score=np.nan,  # set failing folds to NaN (cross_validate doesn't accept "warn")
    )
    fit_sec = float(np.nanmean(cvres["fit_time"]))
    score_sec = float(np.nanmean(cvres["score_time"]))
    t1 = time.time()

    res = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_state": random_state,
        "n_splits": n_splits,
        "groups_key": "file_id",
        "prevalence_positive": float(np.mean(y)),
        "tfidf_ngram_range": cross_cfg.get("tfidf_ngram_range"),
        "tfidf_max_features": cross_cfg.get("tfidf_max_features"),
        "tfidf_min_df": cross_cfg.get("tfidf_min_df"),
        "tfidf_max_df": cross_cfg.get("tfidf_max_df"),
        "vocab_strategy": cross_cfg.get("vocab_strategy"),
        "top_k": cross_cfg.get("top_k"),
        "coverage": cross_cfg.get("coverage"),
        "numeric_scaler": "standard",
        "class_weight": "balanced",
        "model": model_key,
        "params_json": json.dumps(model_params),
        "cv_mean_pr_auc": float(np.nanmean(cvres.get("test_ap", np.array([np.nan])))),
        "cv_std_pr_auc": float(np.nanstd(cvres.get("test_ap", np.array([np.nan])))),
        "cv_mean_f1": float(np.nanmean(cvres.get("test_f1", np.array([np.nan])))),
        "cv_std_f1": float(np.nanstd(cvres.get("test_f1", np.array([np.nan])))),
        "cv_mean_recall": float(np.nanmean(cvres.get("test_recall", np.array([np.nan])))),
        "cv_std_recall": float(np.nanstd(cvres.get("test_recall", np.array([np.nan])))),
        "cv_mean_precision": float(np.nanmean(cvres.get("test_precision", np.array([np.nan])))),
        "cv_std_precision": float(np.nanstd(cvres.get("test_precision", np.array([np.nan])))),
        "cv_mean_roc_auc": float(np.nanmean(cvres.get("test_roc_auc", np.array([np.nan])))),
        "cv_std_roc_auc": float(np.nanstd(cvres.get("test_roc_auc", np.array([np.nan])))),
        "train_time_sec": fit_sec,
        "inference_time_ms_per_1k": float(score_sec * 1000.0),
        "notes": "",
        "wall_clock_sec": float(t1 - t0),
        "cv_mean_accuracy": float(np.nanmean(cvres.get("test_accuracy", np.array([np.nan])))),
        "cv_std_accuracy": float(np.nanstd(cvres.get("test_accuracy", np.array([np.nan])))),
        "cv_mean_bal_accuracy": float(np.nanmean(cvres.get("test_balanced_accuracy", np.array([np.nan])))),
        "cv_std_bal_accuracy": float(np.nanstd(cvres.get("test_balanced_accuracy", np.array([np.nan])))),
        "cv_mean_mcc": float(np.nanmean(cvres.get("test_mcc", np.array([np.nan])))),
        "cv_std_mcc": float(np.nanstd(cvres.get("test_mcc", np.array([np.nan])))),
    }

    # Append
    exp_log_path.parent.mkdir(parents=True, exist_ok=True)
    header = not exp_log_path.exists()
    with open(exp_log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(res.keys()))
        if header:
            writer.writeheader()
        writer.writerow(res)
    return res


def read_grid_csv(path: Path, defaults: List[dict]) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(defaults)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--models", nargs="+", default=["LR", "RF", "MLP"])
    parser.add_argument("--max_rows", type=int, default=999999)
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load & preprocess
    df = preprocess_dataframe(load_all_statements(data_dir))
    X = df[
        [
            "debit_amount", "credit_amount", "balance_amount", "net_amount",
            "abs_amount", "log1p_amount", "hour", "dayofweek", "is_weekend",
            "day", "month", "year", "tx_code", "channel", "description_text",
        ]
    ].copy()
    y = df["fraud_label"].values
    groups = df["file_id"].values

    # Grids (read from out_dir; if missing, fallback defaults)
    grid_cross_path = out_dir / "grid_cross_model.csv"
    grid_lr_path = out_dir / "grid_logreg.csv"
    grid_rf_path = out_dir / "grid_randomforest.csv"
    grid_mlp_path = out_dir / "grid_mlp.csv"

    grid_cross = read_grid_csv(
        grid_cross_path,
        defaults=[
            {
                "tfidf_ngram_range": "1-2",
                "tfidf_max_features": 5000,
                "tfidf_min_df": 5,
                "tfidf_max_df": 0.95,
                "vocab_strategy": "none",
                "top_k": None,
                "coverage": None,
                "n_splits": 5,
            }
        ],
    )
    grid_lr = read_grid_csv(
        grid_lr_path,
        defaults=[{"penalty": "l2", "C": 1.0, "max_iter": 500, "solver": "saga"}],
    )
    grid_rf = read_grid_csv(
        grid_rf_path,
        defaults=[
            {
                "n_estimators": 500,
                "max_depth": None,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "class_weight": "balanced",
            }
        ],
    )
    grid_mlp = read_grid_csv(
        grid_mlp_path,
        defaults=[
            {
                "hidden_layer_sizes": "128",
                "alpha": 1e-4,
                "max_iter": 200,
                "activation": "relu",
                "early_stopping": True,
                "validation_fraction": 0.1,
            }
        ],
    )

    exp_log_path = out_dir / "exp_results.csv"
    print(f"[INFO] Appending results to: {exp_log_path}")

    run_counter = 0
    for i, cross_row in grid_cross.iterrows():
        if run_counter >= args.max_rows:
            break
        n_splits = int(cross_row.get("n_splits", 5))
        preprocessor = build_preprocessor(cross_row.to_dict())

        for model_key in args.models:
            if model_key.upper().startswith("LR"):
                model_grid = grid_lr
            elif model_key.upper().startswith("RF"):
                model_grid = grid_rf
            elif model_key.upper().startswith("MLP"):
                model_grid = grid_mlp
            else:
                print(f"[WARN] Unknown model key in selection: {model_key}")
                continue

            for j, mp in model_grid.iterrows():
                if run_counter >= args.max_rows:
                    break
                model_params = mp.to_dict()
                run_id = f"{int(time.time())}_{i}_{model_key}_{j}"
                print(f"[RUN] cross#{i} {dict(cross_row)} | model={model_key} params={model_params}")
                res = run_cv_and_log(
                    X=X,
                    y=y,
                    groups=groups,
                    preprocessor=preprocessor,
                    model_key=model_key,
                    model_params=model_params,
                    n_splits=n_splits,
                    exp_log_path=exp_log_path,
                    cross_cfg=cross_row.to_dict(),
                    run_id=run_id,
                )
                print(
                    f" -> ACC={res['cv_mean_accuracy']:.4f} (±{res['cv_std_accuracy']:.4f}), "
                    f"BalACC={res['cv_mean_bal_accuracy']:.4f} (±{res['cv_std_bal_accuracy']:.4f}), "
                    f"MCC={res['cv_mean_mcc']:.4f} (±{res['cv_std_mcc']:.4f}), "
                    f"PR-AUC={res['cv_mean_pr_auc']:.4f} (±{res['cv_std_pr_auc']:.4f}), "
                    f"F1={res['cv_mean_f1']:.4f} (±{res['cv_std_f1']:.4f})"
                )
                run_counter += 1

    print("[DONE] Total runs:", run_counter)
    print(f"[OUTPUT] Log at: {exp_log_path}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
