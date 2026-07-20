"""Turn a trained model (or an ensemble) into the arrays the deferral code eats.

Every predict function returns a dict with numpy arrays:
    y_true       (N,)  ground-truth labels
    prob         (N,)  positive-class probability, the score you threshold
    uncertainty  (N,)  higher = defer first
    cohort_idx   (N,)  which cohort each sample came from

collect_logits is separate because temperature scaling is fit on raw logits from
the validation split, before you ever look at the held-out cohort.
"""

from __future__ import annotations

import numpy as np


def collect_logits(model, loader, device: str = "cpu"):
    """Return (logits (N,2), labels (N,)) with dropout OFF. For calibration."""
    import torch

    model.to(device).eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, y, cohort_idx in loader:
            logits = model(imgs.to(device), cohort_idx.to(device))
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def _enable_dropout(model):
    import torch.nn as nn

    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def _entropy_from_prob(p):
    import torch

    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


def predict(model, loader, device: str = "cpu", method: str = "mc_dropout", n_samples: int = 20):
    """MC-dropout (default) or a single deterministic forward pass.

    Uncertainty for mc_dropout is the standard deviation of the positive-class
    probability across dropout samples: a spread-out prediction is an unsure one.
    For a single pass, uncertainty falls back to predictive entropy.
    """
    import torch

    model.to(device).eval()
    if method == "mc_dropout":
        _enable_dropout(model)

    probs, unc, labels, cohorts = [], [], [], []
    with torch.no_grad():
        for imgs, y, cohort_idx in loader:
            imgs = imgs.to(device)
            cidx = cohort_idx.to(device)
            if method == "mc_dropout":
                samples = torch.stack(
                    [torch.softmax(model(imgs, cidx), dim=1)[:, 1] for _ in range(n_samples)],
                    dim=0,
                )  # (T, B)
                mean_p = samples.mean(dim=0)
                u = samples.std(dim=0)
            else:
                p = torch.softmax(model(imgs, cidx), dim=1)[:, 1]
                mean_p = p
                u = _entropy_from_prob(p)
            probs.append(mean_p.cpu().numpy())
            unc.append(u.cpu().numpy())
            labels.append(y.numpy())
            cohorts.append(cohort_idx.numpy())

    return {
        "y_true": np.concatenate(labels),
        "prob": np.concatenate(probs),
        "uncertainty": np.concatenate(unc),
        "cohort_idx": np.concatenate(cohorts),
    }


def ensemble_predict(models, loader, device: str = "cpu"):
    """Deep ensemble: average the members' probabilities; uncertainty is their
    disagreement (std of the positive-class probability across members).
    """
    import torch

    for m in models:
        m.to(device).eval()

    probs, unc, labels, cohorts = [], [], [], []
    with torch.no_grad():
        for imgs, y, cohort_idx in loader:
            imgs = imgs.to(device)
            cidx = cohort_idx.to(device)
            member = torch.stack(
                [torch.softmax(m(imgs, cidx), dim=1)[:, 1] for m in models], dim=0
            )  # (M, B)
            probs.append(member.mean(dim=0).cpu().numpy())
            unc.append(member.std(dim=0).cpu().numpy())
            labels.append(y.numpy())
            cohorts.append(cohort_idx.numpy())

    return {
        "y_true": np.concatenate(labels),
        "prob": np.concatenate(probs),
        "uncertainty": np.concatenate(unc),
        "cohort_idx": np.concatenate(cohorts),
    }
