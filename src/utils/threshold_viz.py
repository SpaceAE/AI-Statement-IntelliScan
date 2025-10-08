# src/utils/threshold_viz.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
)
from sklearn.calibration import calibration_curve

def _to_scores(model, X, force_scores=None):
    """
    คืนค่า probability (class=1) ในช่วง [0,1]
    - ถ้าให้ force_scores มา จะใช้ค่านั้นเลย (เช่น Keras model.predict(...))
    - ถ้าเป็น sklearn estimator: ใช้ predict_proba ถ้ามี, ไม่งั้น scale decision_function
    """
    if force_scores is not None:
        s = np.asarray(force_scores).ravel()
        return np.clip(s, 0.0, 1.0)

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        d = model.decision_function(X)
        d = d.astype(np.float64)
        dmin, dmax = d.min(), d.max()
        return (d - dmin) / (dmax - dmin + 1e-9)
    else:
        raise ValueError("Model must have predict_proba or decision_function, "
                         "or provide force_scores explicitly.")

def _sens_spec(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sens = tp / (tp + fn) if (tp+fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn+fp) > 0 else 0.0
    return sens, spec, (tn, fp, fn, tp)

def _scan_thresholds(scores, y_true, thresholds):
    acc, prec, rec, f1, mcc, sens, spec, costs = [], [], [], [], [], [], [], []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        acc.append(accuracy_score(y_true, y_pred))
        prec.append(precision_score(y_true, y_pred, zero_division=0))
        rec.append(recall_score(y_true, y_pred, zero_division=0))
        f1.append(f1_score(y_true, y_pred, zero_division=0))
        mcc.append(matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0)
        s, p, (tn, fp, fn, tp) = _sens_spec(y_true, y_pred)
        sens.append(s); spec.append(p)
    return {
        "accuracy": np.array(acc),
        "precision": np.array(prec),
        "recall": np.array(rec),
        "f1": np.array(f1),
        "mcc": np.array(mcc),
        "sensitivity": np.array(sens),
        "specificity": np.array(spec),
    }

def _maybe_save(show, save_dir, fname):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, fname)
        plt.savefig(path, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    else:
        plt.close()

def plot_threshold_diagnostics(
    model=None,
    X_val=None,
    y_val=None,
    *,
    scores=None,                # ใช้กรณี Keras: ใส่ prob ของชุด val มาเลย
    thresholds=None,
    show=True,
    save_dir=None,
    file_prefix="val_",
    include_cost=False,
    cost_fp=1.0,
    cost_fn=1.0,
):
    """
    วาดกราฟครบชุดเพื่อเลือก threshold และทำความเข้าใจ trade-off
    - ถ้าเป็น sklearn ให้ส่ง model, X_val, y_val
    - ถ้าเป็น Keras หรือคำนวณ prob เอง ให้ส่ง scores (prob class=1) กับ y_val

    include_cost: ถ้าระบุ True จะวาด Expected cost vs threshold ด้วย (ต้องการ cost_fp, cost_fn)
    """
    assert y_val is not None, "y_val is required"
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    # 1) สร้าง scores
    s = _to_scores(model, X_val, force_scores=scores)

    # 2) คำนวณโค้งตาม threshold
    curves = _scan_thresholds(s, y_val, thresholds)

    # 3) ROC / PR / Calibration (ไม่พึ่ง threshold เดียว)
    fpr, tpr, _ = roc_curve(y_val, s)
    auc_roc = roc_auc_score(y_val, s)
    prec_curve, rec_curve, _ = precision_recall_curve(y_val, s)
    ap = average_precision_score(y_val, s)
    prob_true, prob_pred = calibration_curve(y_val, s, n_bins=10, strategy="quantile")

    # 4) หา best threshold ต่อ metric หลัก
    best = {}
    for k in ["accuracy", "f1", "mcc", "precision", "recall"]:
        idx = int(np.argmax(curves[k]))
        best[k] = (float(thresholds[idx]), float(curves[k][idx]))

    # 5) วาดกราฟ Threshold curves (แยกรูปเพื่ออ่านง่าย)
    for metric_key, ylabel, fname in [
        ("accuracy", "Accuracy", f"{file_prefix}thr_accuracy.png"),
        ("precision", "Precision", f"{file_prefix}thr_precision.png"),
        ("recall", "Recall", f"{file_prefix}thr_recall.png"),
        ("f1", "F1", f"{file_prefix}thr_f1.png"),
        ("mcc", "MCC", f"{file_prefix}thr_mcc.png"),
        ("sensitivity", "Sensitivity (Recall)", f"{file_prefix}thr_sensitivity.png"),
        ("specificity", "Specificity", f"{file_prefix}thr_specificity.png"),
    ]:
        plt.figure()
        plt.plot(thresholds, curves[metric_key])
        bt = best.get(metric_key, None)
        if bt:
            plt.axvline(bt[0], linestyle="--")
        plt.xlabel("Threshold"); plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Threshold")
        plt.grid(True)
        _maybe_save(show, save_dir, fname)

    # 6) ROC
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR")
    plt.ylabel("TPR (Recall)")
    plt.title(f"ROC Curve (AUC = {auc_roc:.4f})")
    plt.grid(True)
    _maybe_save(show, save_dir, f"{file_prefix}roc.png")

    # 7) PR
    plt.figure()
    plt.plot(rec_curve, prec_curve)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall (AP = {ap:.4f})")
    plt.grid(True)
    _maybe_save(show, save_dir, f"{file_prefix}pr.png")

    # 8) Calibration
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("Predicted probability")
    plt.ylabel("True frequency")
    plt.title("Calibration (Reliability) Curve")
    plt.grid(True)
    _maybe_save(show, save_dir, f"{file_prefix}calibration.png")

    # 9) Cost curve (option)
    if include_cost:
        costs = []
        for t in thresholds:
            y_pred = (s >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0,1]).ravel()
            costs.append(cost_fp * fp + cost_fn * fn)
        costs = np.array(costs)
        best_cost_idx = int(np.argmin(costs))
        plt.figure()
        plt.plot(thresholds, costs)
        plt.axvline(thresholds[best_cost_idx], linestyle="--")
        plt.xlabel("Threshold")
        plt.ylabel(f"Expected cost (FP={cost_fp}, FN={cost_fn})")
        plt.title("Cost vs Threshold")
        plt.grid(True)
        _maybe_save(show, save_dir, f"{file_prefix}thr_cost.png")

    return {
        "thresholds": thresholds,
        "curves": curves,
        "best_thresholds": best,
        "roc_auc": float(auc_roc),
        "ap": float(ap),
    }
