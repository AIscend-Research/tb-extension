"""Swappable feature-extractor backbones, via timm.

Backbone-agnosticism is part of the claim: the contribution is the cross-cohort
+ deferral system, not a bespoke network. So the backbone is one config string.
Swap mobilenetv3_small_100 for efficientnet_lite0 or resnet18 and rerun; the
rest of the pipeline does not change.

Defaults lean tiny because the target is edge deployment in a clinic, not a
datacentre. Every backbone here reports its param count and feature dimension so
the "low-compute" claim in the paper is a measured number, not an adjective.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Small, edge-friendly defaults. All exist in timm.
EDGE_BACKBONES = [
    "mobilenetv3_small_100",
    "mobilenetv3_large_100",
    "efficientnet_lite0",
    "efficientnet_b0",
    "resnet18",  # heavier; useful as a stronger reference point
]


class Backbone(nn.Module):
    """Wraps a timm model as a pooled feature extractor.

    forward(x) -> (B, feature_dim). Classification and any domain head live
    outside, in TriageModel, so the same backbone feeds every branch.
    """

    def __init__(
        self,
        name: str = "mobilenetv3_small_100",
        pretrained: bool = True,
        in_chans: int = 1,
        drop_rate: float = 0.2,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm is required for backbones. `pip install timm`, or run on "
                "Kaggle where it is preinstalled."
            ) from e

        # num_classes=0 + global_pool='avg' gives pooled features and no head.
        # drop_rate wires dropout into the backbone so MC-dropout has something
        # to sample from at inference.
        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            in_chans=in_chans,       # 1 = grayscale CXR; timm adapts the stem
            num_classes=0,
            global_pool="avg",
            drop_rate=drop_rate,
        )
        self.feature_dim = self.model.num_features
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_backbone(name: str = "mobilenetv3_small_100", **kwargs) -> Backbone:
    return Backbone(name=name, **kwargs)
