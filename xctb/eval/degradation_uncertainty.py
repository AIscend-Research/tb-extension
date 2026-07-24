"""Does the model's own uncertainty rise with image degradation?

The roadmap's Phase 2 "weak-supervision" idea was to train a confidence head
on labels like "heavy degradation -> high uncertainty is appropriate". The
project instead picked MC-dropout + temperature scaling (see the Phase 1
uncertainty-methods survey), which needs no such training labels. The
same weak label is still useful, just repurposed: as a validation check on
whichever uncertainty the model actually produces (MC-dropout std, ensemble
disagreement, or predictive entropy, all from `xctb/engine/infer.py`).

Torch-free (numpy only), like the rest of `xctb/eval`.
"""

from __future__ import annotations

import numpy as np


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties get the mean rank), the input to Spearman's rho."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # break ties by averaging ranks within equal-value runs
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks


def spearman_correlation(a, b) -> float:
    """Spearman rank correlation in [-1, 1]. NaN if either input is constant
    or the two arrays have fewer than 2 points.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if len(a) < 2:
        return float("nan")
    ra, rb = _rank(a) - _rank(a).mean(), _rank(b) - _rank(b).mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    if denom < 1e-12:
        return float("nan")
    return float((ra * rb).sum() / denom)


def uncertainty_vs_severity(severity, uncertainty) -> dict:
    """Summarise the relationship between degradation severity (the weak label
    from `xctb.data.degradation.severity_to_target_uncertainty`) and the
    model's predicted uncertainty for the same images.

    A model whose uncertainty is honest about degradation, not just about
    label error, should show a clearly positive `spearman_rho`: more degraded
    photos should look less trustworthy to the model, which is the premise the
    "retake photo" deferral message depends on. Near-zero or negative rho means
    the uncertainty signal is not actually responding to image quality and the
    deferral story is not supported by this model yet.
    """
    rho = spearman_correlation(severity, uncertainty)
    return {
        "n": int(len(np.asarray(severity))),
        "spearman_rho": round(rho, 4) if np.isfinite(rho) else float("nan"),
    }
