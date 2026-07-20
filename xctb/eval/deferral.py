"""Deferral evaluation, numpy only. This is the paper's core measurement.

The idea in one line: a trustworthy screener should hand its shakiest cases to a
human instead of guessing. So we sort predictions from most to least certain,
keep the confident ones, and ask how good the kept predictions are as we defer
more of the rest.

Two outputs matter:

  risk-coverage curve   for every coverage level (fraction kept), the error rate
                        on the kept cases. Summarised by AURC (lower is better).
                        This is standard selective-classification territory.

  gap recovery          the metric this project is built around. Take the drop
                        in accuracy going from a same-cohort test to a never-seen
                        cohort. Then ask: if the model defers its most uncertain
                        cases on the new cohort, how much of that drop does it
                        close on the cases it keeps? A model whose uncertainty is
                        honest closes most of the gap by deferring a little. A
                        model that is confidently wrong on the new machine closes
                        almost none, and that failure is exactly what we want to
                        surface before anyone deploys it.

Convention: `uncertainty` is "defer this first" — higher means less trustworthy.
If you hold confidence instead, pass (1 - confidence) or the negated score.
"""

from __future__ import annotations

import numpy as np


def _order_by_confidence(uncertainty):
    """Indices from most confident (lowest uncertainty) to least."""
    u = np.asarray(uncertainty, dtype=float).ravel()
    return np.argsort(u, kind="mergesort")


def risk_coverage_curve(y_true, y_score, uncertainty, threshold: float = 0.5):
    """Return (coverage, risk, aurc).

    coverage[i] = fraction of cases kept after deferring the least confident ones
    risk[i]     = error rate on those kept cases
    aurc        = mean risk over the coverage grid (trapezoidal), lower is better
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    order = _order_by_confidence(uncertainty)

    y_true = y_true[order]
    pred = (y_score[order] >= threshold).astype(int)
    correct = (pred == y_true).astype(float)

    n = len(y_true)
    kept = np.arange(1, n + 1)
    cum_correct = np.cumsum(correct)
    risk = 1.0 - cum_correct / kept          # error among the top-k most confident
    coverage = kept / n

    # Trapezoidal area under the risk-coverage curve. Computed directly so it
    # works across numpy versions (np.trapz was removed in numpy 2.0).
    aurc = float(np.sum((risk[1:] + risk[:-1]) / 2.0 * np.diff(coverage)))
    return coverage, risk, aurc


def accuracy_at_coverage(y_true, y_score, uncertainty, coverage: float, threshold: float = 0.5) -> float:
    """Accuracy on the most-confident `coverage` fraction of cases."""
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1].")
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    order = _order_by_confidence(uncertainty)
    k = max(1, int(round(len(y_true) * coverage)))
    keep = order[:k]
    pred = (y_score[keep] >= threshold).astype(int)
    return float(np.mean(pred == y_true[keep]))


def generalization_gap_recovery(
    acc_in_distribution: float,
    y_true,
    y_score,
    uncertainty,
    coverages=(1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5),
    threshold: float = 0.5,
):
    """How much of the cross-cohort accuracy drop does deferral recover?

    acc_in_distribution : accuracy from the random-split (same-cohort) run, i.e.
                          the optimistic reference number.
    y_true/y_score/uncertainty : predictions on the held-out cohort.

    Returns a dict with:
        gap            = acc_in_distribution - acc_at_full_coverage (the drop)
        rows           = list of per-coverage dicts with keys
                         coverage, deferred, accuracy, recovered, recovery_rate
      where for each coverage c:
        recovered      = accuracy(c) - accuracy(full coverage)
        recovery_rate  = recovered / gap
                         1.0 means the kept cases are back to in-distribution
                         accuracy; > 1.0 means they beat it; near 0 means
                         deferring did not help (a red flag for that cohort).

    If gap <= 0 the held-out cohort was not actually harder, so recovery_rate is
    reported as NaN (there is nothing to recover) and you should just read the
    raw accuracies.
    """
    acc_full = accuracy_at_coverage(y_true, y_score, uncertainty, 1.0, threshold)
    gap = float(acc_in_distribution) - acc_full

    rows = []
    for c in coverages:
        acc_c = accuracy_at_coverage(y_true, y_score, uncertainty, c, threshold)
        recovered = acc_c - acc_full
        rate = (recovered / gap) if gap > 1e-9 else float("nan")
        rows.append(
            {
                "coverage": round(float(c), 4),
                "deferred": round(1.0 - float(c), 4),
                "accuracy": round(float(acc_c), 4),
                "recovered": round(float(recovered), 4),
                "recovery_rate": (round(float(rate), 4) if np.isfinite(rate) else float("nan")),
            }
        )
    return {
        "acc_in_distribution": round(float(acc_in_distribution), 4),
        "acc_cross_cohort": round(float(acc_full), 4),
        "gap": round(float(gap), 4),
        "rows": rows,
    }


def coverage_to_recover(
    acc_in_distribution: float,
    y_true,
    y_score,
    uncertainty,
    target_fraction: float = 0.9,
    threshold: float = 0.5,
    grid=None,
):
    """Smallest coverage (least deferral) that recovers `target_fraction` of the gap.

    Returns (coverage, deferred_fraction, accuracy_at_that_coverage), or None if
    no coverage on the grid reaches the target (or if there is no gap to close).
    """
    acc_full = accuracy_at_coverage(y_true, y_score, uncertainty, 1.0, threshold)
    gap = float(acc_in_distribution) - acc_full
    if gap <= 1e-9:
        return None
    if grid is None:
        grid = np.linspace(1.0, 0.3, 71)  # from keep-all down to keep-30%
    for c in grid:
        acc_c = accuracy_at_coverage(y_true, y_score, uncertainty, float(c), threshold)
        if (acc_c - acc_full) / gap >= target_fraction:
            return float(c), float(1.0 - c), float(acc_c)
    return None
