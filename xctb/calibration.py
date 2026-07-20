"""Confidence calibration, numpy only (no scipy required).

Deferral is only as good as the confidence it sorts by. A model that is
overconfident everywhere will defer the wrong cases. Temperature scaling is the
cheapest fix that works: one scalar T, fit on validation logits, that softens
(or sharpens) the softmax without changing which class wins. So it never hurts
accuracy, only the confidence numbers that feed the deferral policy.

Fit T on the validation split of the *seen* cohorts, then apply it to the
held-out cohort at test time. See scripts/evaluate.py for the wiring.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    if logits.ndim == 1:  # treat a single logit as the positive-class score
        logits = np.stack([np.zeros_like(logits), logits], axis=1)
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, labels: np.ndarray, T: float) -> float:
    p = softmax(logits / T)
    idx = np.arange(len(labels))
    picked = np.clip(p[idx, labels.astype(int)], 1e-12, 1.0)
    return float(-np.mean(np.log(picked)))


def fit_temperature(logits, labels, bounds=(0.05, 20.0), iters: int = 60) -> float:
    """Golden-section search for the T that minimises validation NLL."""
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels).astype(int).ravel()
    lo, hi = bounds
    gr = (np.sqrt(5) - 1) / 2
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc, fd = _nll(logits, labels, c), _nll(logits, labels, d)
    for _ in range(iters):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = _nll(logits, labels, c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = _nll(logits, labels, d)
    return float((lo + hi) / 2)


def apply_temperature(logits, T: float) -> np.ndarray:
    """Return calibrated positive-class probabilities."""
    return softmax(np.asarray(logits, dtype=float) / T)[:, 1]


def expected_calibration_error(probs, labels, n_bins: int = 15) -> float:
    """ECE over confidence bins. probs are positive-class probabilities."""
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels).astype(int).ravel()
    conf = np.where(probs >= 0.5, probs, 1.0 - probs)  # confidence in the winner
    pred = (probs >= 0.5).astype(int)
    correct = (pred == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            acc = correct[in_bin].mean()
            avg_conf = conf[in_bin].mean()
            ece += (in_bin.sum() / n) * abs(avg_conf - acc)
    return float(ece)
