"""Lightweight domain-generalization losses.

These are deliberately small reimplementations rather than a dependency on a
full benchmark suite, so the codebase stays edge-focused and easy to read. The
reference implementations in DomainBed and DeepDG are worth reading if you want
the fuller variants; links in docs/ONBOARDING.md.

DANN is not here because it is not a standalone loss: it is the domain head plus
the gradient-reversal layer, driven from the training loop. See
xctb.models.grl and engine/train.py.
"""

from __future__ import annotations

import torch


def coral_loss(features_by_cohort: list[torch.Tensor]) -> torch.Tensor:
    """Deep CORAL (Sun & Saenko, 2016): align feature mean and covariance across
    cohorts. Returns the mean pairwise alignment penalty over all cohort pairs.

    features_by_cohort: list of (n_i, d) tensors, one per cohort present in the
    batch. Cohorts with fewer than 2 samples are skipped (covariance needs 2+).
    """
    feats = [f for f in features_by_cohort if f.size(0) >= 2]
    if len(feats) < 2:
        return torch.zeros((), device=feats[0].device if feats else "cpu")

    def mean_cov(f):
        mu = f.mean(dim=0, keepdim=True)
        fc = f - mu
        cov = (fc.t() @ fc) / (f.size(0) - 1)
        return mu.squeeze(0), cov

    stats = [mean_cov(f) for f in feats]
    d = feats[0].size(1)
    total = 0.0
    pairs = 0
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            mu_i, cov_i = stats[i]
            mu_j, cov_j = stats[j]
            mean_term = torch.sum((mu_i - mu_j) ** 2)
            cov_term = torch.sum((cov_i - cov_j) ** 2) / (4 * d * d)
            total = total + mean_term + cov_term
            pairs += 1
    return total / max(pairs, 1)


def irm_penalty(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """IRM penalty (Arjovsky et al., 2019): squared gradient of the per-cohort
    loss w.r.t. a dummy scale of 1.0. Call once per cohort and average.
    """
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(logits * scale, targets)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)
