"""Training loop.

One loop covers the plain baseline and every domain-invariance variant, chosen
by cfg["dg_method"]:

    none    cross-entropy only. This is the baseline that produces the
            generalization gap you are trying to explain.
    coral   add feature mean/covariance alignment across the cohorts in a batch.
    dann    add a domain classifier through a gradient-reversal layer.
    irm     add the IRM gradient penalty per cohort.

Model selection uses the validation split of the *seen* cohorts (never the
held-out one), because at deployment you would not have labels from the new
clinic to tune on.
"""

from __future__ import annotations

import copy

import numpy as np

from xctb.losses.dg import coral_loss, irm_penalty
from xctb.models.grl import dann_lambda


def _features_by_cohort(features, cohort_idx):
    groups = []
    for c in cohort_idx.unique():
        groups.append(features[cohort_idx == c])
    return groups


def _dg_loss(model, features, logits, labels, cohort_idx, cfg, step, total_steps):
    import torch
    import torch.nn.functional as F

    method = str(cfg.get("dg_method", "none")).lower()
    weight = float(cfg.get("dg_weight", 1.0))
    if method == "none" or weight == 0.0:
        return torch.zeros((), device=logits.device)

    if method == "coral":
        return weight * coral_loss(_features_by_cohort(features, cohort_idx))

    if method == "dann":
        lambd = dann_lambda(step, total_steps)
        domain_logits = model.domain_logits(features, lambd)
        return weight * F.cross_entropy(domain_logits, cohort_idx)

    if method == "irm":
        penalties = []
        for c in cohort_idx.unique():
            mask = cohort_idx == c
            penalties.append(irm_penalty(logits[mask], labels[mask]))
        return weight * torch.stack(penalties).mean()

    raise ValueError(f"unknown dg_method {method!r}")


def train_one_run(model, train_loader, val_loader, cfg: dict, device: str = "cpu"):
    """Train a model and return (best_model, history).

    history is a list of per-epoch dicts (train loss, val accuracy/AUROC).
    The returned model has the best validation weights loaded.
    """
    import torch
    import torch.nn.functional as F

    from xctb.eval.metrics import binary_report

    model.to(device)
    epochs = int(cfg.get("epochs", 20))
    lr = float(cfg.get("lr", 3e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = max(epochs * max(len(train_loader), 1), 1)
    step = 0
    best_score = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for imgs, labels, cohort_idx in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            cohort_idx = cohort_idx.to(device)

            logits, features = model(imgs, cohort_idx, return_features=True)
            loss = F.cross_entropy(logits, labels)
            loss = loss + _dg_loss(model, features, logits, labels, cohort_idx, cfg, step, total_steps)

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            step += 1

        val = _evaluate(model, val_loader, device)
        score = val["auroc"] if np.isfinite(val["auroc"]) else val["accuracy"]
        history.append({"epoch": epoch, "train_loss": running / max(len(train_loader), 1), **val})
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history


def _evaluate(model, loader, device):
    import torch

    from xctb.eval.metrics import binary_report

    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for imgs, y, cohort_idx in loader:
            logits = model(imgs.to(device), cohort_idx.to(device))
            p = torch.softmax(logits, dim=1)[:, 1]
            scores.append(p.cpu().numpy())
            labels.append(y.numpy())
    import numpy as np

    return binary_report(np.concatenate(labels), np.concatenate(scores))
