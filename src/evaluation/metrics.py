"""Recognition quality and paired-seed comparison metrics (Slide 10)."""

from __future__ import annotations

import numpy as np


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    activity_names: list[str],
) -> dict[str, object]:
    """Compute overall accuracy, macro-F1, per-class F1, and confusion matrix in pure NumPy."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    K = len(activity_names)

    acc = float(np.mean(y_true == y_pred))

    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < K and 0 <= p < K:
            cm[t, p] += 1

    per_class_f1: dict[str, float] = {}
    f1_list: list[float] = []
    for c in range(K):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        denom = 2.0 * tp + fp + fn
        f1_c = (2.0 * tp / denom) if denom > 0 else 0.0
        per_class_f1[activity_names[c]] = f1_c
        f1_list.append(f1_c)

    macro_f1 = float(np.mean(f1_list))
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
    }
