"""Plain binary-classification metrics, numpy only.

TB screening cares far more about sensitivity than raw accuracy: a missed TB
case is much worse than a false alarm sent for a second read. So binary_report
returns sensitivity and specificity separately and lets you set the operating
threshold, rather than hiding everything behind one accuracy number.
"""

from __future__ import annotations

import numpy as np


def _as_arrays(y_true, y_score):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}")
    return y_true, y_score


def accuracy(y_true, y_score, threshold: float = 0.5) -> float:
    y_true, y_score = _as_arrays(y_true, y_score)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_score >= threshold).astype(int) == y_true))


def sensitivity_specificity(y_true, y_score, threshold: float = 0.5) -> tuple[float, float]:
    """Return (sensitivity, specificity) = (TP rate, TN rate)."""
    y_true, y_score = _as_arrays(y_true, y_score)
    pred = (y_score >= threshold).astype(int)
    pos = y_true == 1
    neg = y_true == 0
    sens = float(np.mean(pred[pos] == 1)) if pos.any() else float("nan")
    spec = float(np.mean(pred[neg] == 0)) if neg.any() else float("nan")
    return sens, spec


def auroc(y_true, y_score) -> float:
    """Area under the ROC curve. Uses sklearn when present, else a rank formula."""
    y_true, y_score = _as_arrays(y_true, y_score)
    if len(set(y_true.tolist())) < 2:
        return float("nan")  # AUROC is undefined with one class present
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        # Mann-Whitney U statistic with average ranks for ties.
        order = np.argsort(y_score, kind="mergesort")
        ranks = np.empty(len(y_score), dtype=float)
        ranks[order] = np.arange(1, len(y_score) + 1)
        # average tied ranks
        _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))
        rank_sum_pos = float(np.sum(ranks[y_true == 1]))
        return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def binary_report(y_true, y_score, threshold: float = 0.5) -> dict:
    sens, spec = sensitivity_specificity(y_true, y_score, threshold)
    return {
        "n": int(len(np.asarray(y_true).ravel())),
        "accuracy": accuracy(y_true, y_score, threshold),
        "sensitivity": sens,
        "specificity": spec,
        "auroc": auroc(y_true, y_score),
        "threshold": threshold,
    }
