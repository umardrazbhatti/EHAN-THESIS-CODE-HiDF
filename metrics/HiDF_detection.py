"""
metrics/detection.py — AUC-ROC, AUC-PR, F1 and per-class metrics for binary deepfake detection.
Handles the edge case where only one class is present in y_true (returns nan/0).
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, roc_curve,
)
import warnings


def compute_detection_metrics(probs, labels) -> dict:
    """
    Compute detection metrics.  Guards the single-class case.

    Returns
    -------
    dict with keys:
      auc_roc, auc_pr,
      f1_at_0.5, balanced_accuracy, macro_f1,
      real_accuracy, fake_accuracy,
      optimal_threshold, f1_at_optimal, balanced_accuracy_at_optimal
    """
    labels = np.array(labels, dtype=int)
    probs  = np.array(probs,  dtype=float)

    unique = np.unique(labels)
    if len(unique) < 2:
        warnings.warn(
            f"Only class(es) {unique.tolist()} present in labels. "
            "Fix the dataset loader — both classes required. "
            "AUC-ROC and AUC-PR are undefined; returning NaN."
        )
        return {
            "auc_roc":                      float("nan"),
            "auc_pr":                       float("nan"),
            "f1_at_0.5":                    0.0,
            "balanced_accuracy":            0.0,
            "macro_f1":                     0.0,
            "real_accuracy":                0.0,
            "fake_accuracy":                0.0,
            "optimal_threshold":            0.5,
            "f1_at_optimal":               0.0,
            "balanced_accuracy_at_optimal": 0.0,
        }

    preds = (probs >= 0.5).astype(int)

    # Per-class masks
    real_mask = labels == 0
    fake_mask = labels == 1

    # Per-class accuracy at threshold 0.5
    real_accuracy = float((preds[real_mask] == 0).sum()) / max(int(real_mask.sum()), 1)
    fake_accuracy = float((preds[fake_mask] == 1).sum()) / max(int(fake_mask.sum()), 1)

    # Balanced accuracy = 0.5*(TPR + TNR) at 0.5
    # TPR = fake_accuracy (fakes caught), TNR = real_accuracy (reals correct)
    balanced_accuracy = 0.5 * (fake_accuracy + real_accuracy)

    # Optimal threshold (maximises balanced accuracy on the ROC curve)
    fpr_arr, tpr_arr, thresholds = roc_curve(labels, probs)
    balanced_arr      = 0.5 * (tpr_arr + (1.0 - fpr_arr))
    opt_idx           = int(np.argmax(balanced_arr))
    optimal_threshold = float(thresholds[opt_idx])

    preds_opt      = (probs >= optimal_threshold).astype(int)
    f1_at_optimal  = float(f1_score(labels, preds_opt, zero_division=0))
    real_acc_opt   = float((preds_opt[real_mask] == 0).sum()) / max(int(real_mask.sum()), 1)
    fake_acc_opt   = float((preds_opt[fake_mask] == 1).sum()) / max(int(fake_mask.sum()), 1)
    bal_acc_opt    = 0.5 * (real_acc_opt + fake_acc_opt)

    return {
        "auc_roc":                      float(roc_auc_score(labels, probs)),
        "auc_pr":                       float(average_precision_score(labels, probs)),
        "f1_at_0.5":                    float(f1_score(labels, preds, zero_division=0)),
        "balanced_accuracy":            balanced_accuracy,
        "macro_f1":                     float(f1_score(labels, preds, average="macro", zero_division=0)),
        "real_accuracy":                real_accuracy,
        "fake_accuracy":                fake_accuracy,
        "optimal_threshold":            optimal_threshold,
        "f1_at_optimal":               f1_at_optimal,
        "balanced_accuracy_at_optimal": bal_acc_opt,
    }


class DetectionMetrics:
    @staticmethod
    def compute(probs, labels) -> dict:
        return compute_detection_metrics(probs, labels)
