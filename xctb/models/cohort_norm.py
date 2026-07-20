"""Cohort-conditional feature modulation. The cross-field extension.

Borrowed framing from wireless channel estimation: treat each cohort's imaging
signature (machine, contrast profile, protocol) as a distinct channel with its
own characteristics, and adapt features per channel instead of forcing one
invariant representation for all of them. Here that adaptation is a FiLM-style
affine modulation (Perez et al., 2018): a per-cohort embedding produces a scale
and shift applied to the pooled features.

Why this might beat plain domain-invariance: invariance throws away anything
cohort-specific, including signal that happens to correlate with the machine.
Conditioning keeps a per-cohort knob instead. The open question the ablation
should answer is what to do at test time on a *never-seen* cohort, where you
have no learned embedding. The stub below defaults such cohorts to the mean
training embedding; trying alternatives (a small adapter fit on a handful of
unlabeled target images, or averaging the k nearest cohort embeddings) is a
concrete piece of novel work for whoever picks this up. See docs/ONBOARDING.md.

This is a starting point, not a finished method. Marked TODO where it matters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CohortFiLM(nn.Module):
    def __init__(self, feature_dim: int, num_cohorts: int, embed_dim: int = 16):
        super().__init__()
        self.num_cohorts = num_cohorts
        self.embed = nn.Embedding(num_cohorts, embed_dim)
        self.to_scale = nn.Linear(embed_dim, feature_dim)
        self.to_shift = nn.Linear(embed_dim, feature_dim)
        # Start near identity: scale ~ 1, shift ~ 0.
        nn.init.zeros_(self.to_scale.weight)
        nn.init.ones_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def mean_embedding(self) -> torch.Tensor:
        return self.embed.weight.mean(dim=0, keepdim=True)

    def forward(self, features: torch.Tensor, cohort_idx: torch.Tensor | None) -> torch.Tensor:
        if cohort_idx is None:
            # Unseen cohort at test time: fall back to the average training
            # cohort. TODO(extension): replace with a target-adaptive estimate.
            emb = self.mean_embedding().expand(features.size(0), -1)
        else:
            emb = self.embed(cohort_idx)
        scale = self.to_scale(emb)
        shift = self.to_shift(emb)
        return features * scale + shift
