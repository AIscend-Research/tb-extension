"""Baseline TB classifier: a standard backbone + a dropout head.

Why a backbone and not TB-Net out of the box? TB-Net's published code is
TensorFlow 1.15 / checkpoint format (see models/tbnet.py). To let the whole team
run end-to-end on day one, the default runnable model is a torchvision/timm CNN
(DenseNet-121 is the CheXNet-style default for chest X-rays). Reproducing TB-Net's
attention-condenser architecture is a first-class task, tracked in tbnet.py, and
drops in behind the same interface.

The head keeps dropout *active at inference* when asked, which is what makes
MC-dropout uncertainty possible (see models/uncertainty.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.manifest import NUM_CLINIC_SLOTS
from .clinic_film import ClinicFiLM
from .grl import grad_reverse


def _make_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Return (feature_extractor, feature_dim). Tries timm, falls back to torchvision."""
    timm_error: Exception | None = None
    try:
        import timm

        model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
        return model, model.num_features
    except Exception as exc:
        timm_error = exc
        from torchvision import models as tvm

        # Explicit table, and an error for anything not in it. The previous
        # fallback was `densenet* -> densenet121, everything else -> resnet50`,
        # which meant a config asking for resnet18 silently trained a resnet50
        # whenever timm was missing or could not reach its weight host -- the run
        # succeeds, the log says resnet18, and the parameter count in
        # benchmark_efficiency.py is for a different network than the one that
        # produced the accuracy.
        known = {
            "densenet121": (tvm.densenet121, tvm.DenseNet121_Weights),
            "resnet18": (tvm.resnet18, tvm.ResNet18_Weights),
            "resnet34": (tvm.resnet34, tvm.ResNet34_Weights),
            "resnet50": (tvm.resnet50, tvm.ResNet50_Weights),
        }
        if name not in known:
            raise ValueError(
                f"backbone {name!r} is not available: timm could not provide it "
                f"and the torchvision fallback knows only {sorted(known)}. "
                "Install timm, or pick one of those.") from timm_error
        ctor, weights = known[name]
        net = ctor(weights=weights.DEFAULT if pretrained else None)
        if hasattr(net, "classifier"):
            dim = net.classifier.in_features
            net.classifier = nn.Identity()
        else:
            dim = net.fc.in_features
            net.fc = nn.Identity()
        return net, dim


def expand_first_conv(backbone: nn.Module, in_channels: int) -> nn.Module:
    """Widen the stem convolution to `in_channels`, zero-initialising the new ones.

    Zero-init is the point, not a default. With the extra kernel at zero the
    network's output does not depend on the new channel at all at initialisation
    -- feed it noise, zeros or a constant and the logit is bit for bit the same,
    which `tests/test_physics_training.py` pins. So the physics arm starts as the
    identical function to the same network with the channel removed, and any
    divergence afterwards is something the channel bought rather than a different
    starting point. Copying the mean of the pretrained RGB weights
    into the new channel instead -- the usual recipe -- would perturb every
    prediction from step zero and confound "the physics helped" with "the stem
    was reinitialised".

    The pretrained weights for the first three channels are left untouched, so
    ImageNet initialisation survives.
    """
    conv = None
    for m in backbone.modules():
        if isinstance(m, nn.Conv2d):
            conv = m
            break
    if conv is None:
        raise ValueError("no Conv2d in the backbone to widen")
    if conv.in_channels == in_channels:
        return backbone
    if conv.in_channels != 3:
        raise ValueError(f"expected a 3-channel stem, found {conv.in_channels}")

    wider = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size,
                      stride=conv.stride, padding=conv.padding,
                      dilation=conv.dilation, groups=conv.groups,
                      bias=conv.bias is not None)
    with torch.no_grad():
        wider.weight.zero_()
        wider.weight[:, :3] = conv.weight
        if conv.bias is not None:
            wider.bias.copy_(conv.bias)

    parent, attr = None, None
    for _name, module in backbone.named_modules():
        for cname, child in module.named_children():
            if child is conv:
                parent, attr = module, cname
    if parent is None:
        raise ValueError("could not locate the stem convolution's parent module")
    setattr(parent, attr, wider)
    return backbone


class TBClassifier(nn.Module):
    """Backbone -> dropout -> logit(s). Optionally a second head for uncertainty,
    a domain-adversarial head (DANN), and clinic-conditional FiLM modulation.

    Outputs a dict:
        logit       : (B,) raw score for the TB class
        features    : (B, D) pooled backbone features, pre-dropout. The
                      domain-generalization losses in losses/dg.py operate on
                      these, so the training loop needs them out of one forward
                      pass rather than a second one.
        uncertainty : (B,) optional predicted 'appropriate uncertainty' in [0,1]
                      (only if with_uncertainty_head=True)

    `clinic_idx` is optional everywhere: the whole eval path calls `model(x)`
    with no clinic label, which is correct under LOCO -- the held-out clinic has
    no learned embedding, and FiLM falls back to the mean training clinic.
    """

    def __init__(
        self,
        backbone: str = "densenet121",
        pretrained: bool = True,
        dropout: float = 0.3,
        with_uncertainty_head: bool = True,
        num_clinics: int = NUM_CLINIC_SLOTS,
        with_domain_head: bool = False,
        with_clinic_film: bool = False,
        in_channels: int = 3,
    ):
        super().__init__()
        self.backbone, feat = _make_backbone(backbone, pretrained)
        self.in_channels = int(in_channels)
        if self.in_channels != 3:
            self.backbone = expand_first_conv(self.backbone, self.in_channels)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat, 1)
        self.with_uncertainty_head = with_uncertainty_head
        if with_uncertainty_head:
            self.uncertainty_head = nn.Sequential(nn.Linear(feat, 64), nn.ReLU(), nn.Linear(64, 1))

        self.film = ClinicFiLM(feat, num_clinics) if with_clinic_film else None
        self.domain_head = None
        if with_domain_head:
            self.domain_head = nn.Sequential(
                nn.Linear(feat, feat // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(feat // 2, num_clinics),
            )

    def forward(self, x: torch.Tensor, clinic_idx: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        if self.film is not None:
            feat = self.film(feat, clinic_idx)
        dropped = self.dropout(feat)
        out = {"logit": self.classifier(dropped).squeeze(-1), "features": feat}
        if self.with_uncertainty_head:
            out["uncertainty"] = torch.sigmoid(self.uncertainty_head(dropped).squeeze(-1))
        return out

    def domain_logits(self, features: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        """Clinic prediction through the gradient-reversal layer (DANN)."""
        if self.domain_head is None:
            raise RuntimeError("Model was built without a domain head (dg.method is not 'dann').")
        return self.domain_head(grad_reverse(features, lambd))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x)["logit"])


def build_model(cfg: dict) -> TBClassifier:
    m = cfg.get("model", {})
    dg = str(cfg.get("dg", {}).get("method", "none")).lower()
    return TBClassifier(
        backbone=m.get("backbone", "densenet121"),
        pretrained=m.get("pretrained", True),
        dropout=m.get("dropout", 0.3),
        with_uncertainty_head=m.get("with_uncertainty_head", True),
        num_clinics=m.get("num_clinics", NUM_CLINIC_SLOTS),
        with_domain_head=(dg == "dann"),
        with_clinic_film=bool(m.get("clinic_film", False)),
        in_channels=int(m.get("in_channels", 3)),
    )
