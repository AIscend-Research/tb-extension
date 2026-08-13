"""Learned phone-camera degradation model (Phase 2 extension).

`degradation.py` is physics-parametrized: each op (blur, glare, shadow, angle,
JPEG, downscale) has a hand-picked severity->parameter mapping. That's
interpretable and needs no data, but it can only ever be as realistic as the
hand-tuning. This module is the other end of the ablation: a small generator
network that *learns* the clean-film -> phone-photo mapping from example real
recaptures, so its artifacts are whatever the real capture process actually
produces (including correlations between blur/glare/shadow the physics pipeline
applies independently and can't reproduce).

Unpaired by design. Getting *paired* (same film, clean scan + phone photo) data
requires physically recapturing existing film, which this project does not have
(see `data/real_recapture/README.md` for the collection protocol and why it's a
placeholder for now). So this is trained the way unpaired image-to-image
translation is normally trained: a generator that tries to make clean images
"look like" the real-photo domain, judged by a discriminator trained to tell them
apart, plus an identity/reconstruction penalty so the generator perturbs the image
instead of replacing it (there's no cycle-consistency partner network here since
we only need the one direction, clean -> photo; that keeps this small enough to
train on a laptop and to reason about, at the cost of the stability cycle-GANs get
from the round trip).

Honest status: the architecture and training loop are real and exercised by
`scripts/ablate_degradation.py` and the smoke test below, but there is currently
no real recapture set to train it on (see the README). Until one exists, treat
any `LearnedDegrader` checkpoint as a proof the pipeline runs, not as validated
against real phone-camera artifacts -- that validation is the point of the
ablation once real data lands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def _require_torch():
    if torch is None:
        raise ImportError("degradation_learned needs torch. `pip install -e .`")


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(ch, affine=True)
        self.norm2 = nn.InstanceNorm2d(ch, affine=True)

    def forward(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return x + h


class Generator(nn.Module):
    """Clean image (+ a severity scalar broadcast as an extra channel) -> degraded image.

    Deliberately tiny (a handful of conv layers, ~severity channels) -- this only
    needs to learn a *capture-process* perturbation, not a general image model.
    Outputs a residual added to the input and clamped to [0,1], so at severity 0
    (an all-zero conditioning channel) it can learn to be close to identity.
    """

    def __init__(self, in_channels: int = 1, base: int = 16, n_blocks: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels + 1, base, 7, padding=3), nn.InstanceNorm2d(base, affine=True), nn.ReLU()
        )
        self.blocks = nn.Sequential(*[ResBlock(base) for _ in range(n_blocks)])
        self.head = nn.Conv2d(base, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, severity: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        sev_map = severity.view(b, 1, 1, 1).expand(b, 1, h, w)
        feat = self.stem(torch.cat([x, sev_map], dim=1))
        feat = self.blocks(feat)
        residual = torch.tanh(self.head(feat))
        return torch.clamp(x + residual * severity.view(b, 1, 1, 1), 0.0, 1.0)


class PatchDiscriminator(nn.Module):
    """Small PatchGAN: classifies overlapping patches real-photo vs. generated, not the whole image.

    Patch-level judgments are what make this trainable on a handful of real
    recaptures -- one photo gives many patch samples instead of one label.
    """

    def __init__(self, in_channels: int = 1, base: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base * 2, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base * 4, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base * 4, 1, 4, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 1, h', w') patch logits, no sigmoid (BCEWithLogits at the loss site)


@dataclass
class LearnedDegrader:
    """Wraps a Generator with the same call convention as `SmartphoneDegradation`.

    So the ablation script and, eventually, the Dataset can swap physics-based and
    learned degradation behind one interface. `severity` still means the same
    thing (0 = untouched) even though the generator, not a hand-written formula,
    decides what "untouched" and "maximally degraded" look like.
    """

    generator: Generator = None
    device: str = "cpu"

    def __post_init__(self):
        _require_torch()
        if self.generator is None:
            self.generator = Generator()
        self.generator = self.generator.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, img: np.ndarray, severity: float) -> np.ndarray:
        x = torch.from_numpy(img.astype(np.float32) / 255.0)
        x = x.unsqueeze(0) if x.ndim == 2 else x.permute(2, 0, 1)
        x = x.unsqueeze(0).to(self.device)
        sev = torch.tensor([severity], dtype=torch.float32, device=self.device)
        out = self.generator(x, sev)[0]
        out = (out.clamp(0, 1) * 255.0).byte().cpu().numpy()
        return out[0] if out.shape[0] == 1 else out.transpose(1, 2, 0)

    def save(self, path: str) -> None:
        torch.save(self.generator.state_dict(), path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> LearnedDegrader:
        gen = Generator()
        gen.load_state_dict(torch.load(path, map_location=device))
        return cls(generator=gen, device=device)


def train_learned_degradation(
    clean_images: list[np.ndarray],
    real_photo_images: list[np.ndarray],
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 2e-4,
    identity_weight: float = 10.0,
    device: str = "cpu",
    seed: int = 0,
) -> LearnedDegrader:
    """Unpaired adversarial training: clean_images (source domain) -> real_photo_images (target domain).

    Loss = adversarial (fool the discriminator into calling generated images
    real) + `identity_weight` * L1 reconstruction at severity 0 (forces the
    generator to be near-identity when asked for "no degradation," which the
    physics pipeline gets for free but a learned generator has to be told).

    Needs real recaptured photos to be meaningful -- see `data/real_recapture/`.
    Runs fine on the tiny synthetic images the smoke test uses (proves the loop
    is wired correctly); the *output* is only as realistic as `real_photo_images`.
    """
    _require_torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    gen = Generator().to(device)
    disc = PatchDiscriminator().to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    def _batch(images: list[np.ndarray], n: int) -> torch.Tensor:
        idx = rng.integers(0, len(images), size=n)
        arrs = [np.asarray(images[i], dtype=np.float32) / 255.0 for i in idx]
        t = torch.from_numpy(np.stack(arrs))[:, None, :, :]
        return t.to(device)

    for _epoch in range(epochs):
        clean = _batch(clean_images, batch_size)
        real_photo = _batch(real_photo_images, batch_size)
        severity = torch.rand(batch_size, device=device)

        fake = gen(clean, severity)

        # discriminator step
        opt_d.zero_grad()
        d_real = disc(real_photo)
        d_fake = disc(fake.detach())
        d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        d_loss.backward()
        opt_d.step()

        # generator step
        opt_g.zero_grad()
        d_fake_for_g = disc(fake)
        adv_loss = bce(d_fake_for_g, torch.ones_like(d_fake_for_g))
        identity = gen(clean, torch.zeros(batch_size, device=device))
        id_loss = F.l1_loss(identity, clean)
        g_loss = adv_loss + identity_weight * id_loss
        g_loss.backward()
        opt_g.step()

    return LearnedDegrader(generator=gen, device=device)
