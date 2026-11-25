"""
Computes standard biometric presentation attack and deepfake metrics:
- ISO/IEC 30107-3 PAD Metrics: APCER, BPCER, ACER
- Classification Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
"""
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_biometric_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    apcer = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    bpcer = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    acer = 0.5 * (apcer + bpcer)

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0

    return {
        "threshold": threshold,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(roc_auc, 4),
        "APCER": round(apcer, 4),
        "BPCER": round(bpcer, 4),
        "ACER": round(acer, 4),
        "confusion_matrix": {
            "TN_bona_fide_accepted": int(tn),
            "FP_bona_fide_rejected": int(fp),
            "FN_attack_accepted": int(fn),
            "TP_attack_rejected": int(tp),
        },
    }
