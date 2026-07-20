"""The triage model: one backbone feeding a TB classifier and, optionally, a
domain-adversarial head and a cohort-conditional modulation layer.

Uncertainty here is MC-dropout: keep dropout switched on at inference, run the
image through several times, and read the spread of the predictions. Wide spread
means the model is unsure, which is exactly the signal the deferral policy sorts
on. Deep ensembles are the other option and usually calibrate better; that is a
training-script concern (train N seeds), see engine/infer.py and scripts/.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xctb.models.backbones import create_backbone
from xctb.models.grl import grad_reverse
from xctb.models.cohort_norm import CohortFiLM


def _enable_dropout(module: nn.Module) -> None:
    """Put only the dropout layers into training mode (for MC-dropout)."""
    for m in module.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


class TriageModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "mobilenetv3_small_100",
        num_cohorts: int = 4,
        num_classes: int = 2,
        in_chans: int = 1,
        pretrained: bool = True,
        drop_rate: float = 0.2,
        use_domain_head: bool = False,
        use_cohort_film: bool = False,
    ):
        super().__init__()
        self.backbone = create_backbone(
            backbone_name, pretrained=pretrained, in_chans=in_chans, drop_rate=drop_rate
        )
        dim = self.backbone.feature_dim

        self.film = CohortFiLM(dim, num_cohorts) if use_cohort_film else None
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(dim, num_classes)

        self.domain_head = None
        if use_domain_head:
            self.domain_head = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_rate),
                nn.Linear(dim // 2, num_cohorts),
            )

    def features(self, x: torch.Tensor, cohort_idx: torch.Tensor | None = None) -> torch.Tensor:
        f = self.backbone(x)
        if self.film is not None:
            f = self.film(f, cohort_idx)
        return f

    def forward(self, x, cohort_idx=None, return_features: bool = False):
        f = self.features(x, cohort_idx)
        logits = self.classifier(self.dropout(f))
        if return_features:
            return logits, f
        return logits

    def domain_logits(self, features: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        """Cohort prediction through the gradient-reversal layer (DANN)."""
        if self.domain_head is None:
            raise RuntimeError("Model was built without a domain head (use_domain_head=False).")
        return self.domain_head(grad_reverse(features, lambd))

    @torch.no_grad()
    def mc_predict(self, x, cohort_idx=None, n_samples: int = 20) -> torch.Tensor:
        """Return per-sample class probabilities averaged over MC-dropout passes.

        Also usable for uncertainty: call mc_logits if you want the full sample
        stack to compute predictive entropy or variance (see engine/infer.py).
        """
        was_training = self.training
        self.eval()
        _enable_dropout(self)  # dropout back on, everything else stays in eval
        probs = []
        for _ in range(n_samples):
            logits = self.forward(x, cohort_idx)
            probs.append(torch.softmax(logits, dim=1))
        out = torch.stack(probs, dim=0).mean(dim=0)
        if was_training:
            self.train()
        return out


def build_model(cfg: dict) -> TriageModel:
    """Construct a TriageModel from a config dict (see configs/base.yaml)."""
    dg = str(cfg.get("dg_method", "none")).lower()
    return TriageModel(
        backbone_name=cfg.get("backbone", "mobilenetv3_small_100"),
        num_cohorts=int(cfg.get("num_cohorts", 4)),
        in_chans=int(cfg.get("in_chans", 1)),
        pretrained=bool(cfg.get("pretrained", True)),
        drop_rate=float(cfg.get("drop_rate", 0.2)),
        use_domain_head=(dg == "dann"),
        use_cohort_film=bool(cfg.get("use_cohort_film", False)),
    )
