"""Synthetic smartphone-capture degradation for X-ray film re-photography.

Rural clinics often work from a smartphone photo of a printed/lightbox film,
not a direct digital export. This module fabricates that gap: blur (motion and
defocus), glare (a specular highlight off the lightbox), shadow (non-uniform
lighting), rotation/angle, JPEG compression, and reduced resolution. Every
function takes a continuous `severity` in [0, 1] rather than a binary
"degraded / not degraded" flag, so the uncertainty analysis in
`xctb/eval/degradation_uncertainty.py` can ask a fine-grained question
("does trust fall off smoothly as quality drops?") instead of a binary one.

Torch-free by design (PIL + numpy only), like the rest of `xctb/data`, so it
can run in the same lightweight environment as manifest/splits.

Strategy comparison (the "which simulation method matches real smartphone
artifacts" ablation from the roadmap): `DEGRADATION_STRATEGIES` lists what is
implemented (`simple`, `full`) and what is deliberately stubbed
(`gan`, `rephoto`) because it needs assets we do not have yet (a trained phone-
camera model, or actual re-photographed prints). Calling a stubbed strategy
raises `NotImplementedError` with what would be needed, rather than silently
falling back to something else.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

# --------------------------------------------------------------------------- #
# Individual degradations. Each takes a PIL "L" image, a severity in [0, 1],
# and a numpy Generator, and returns a new PIL "L" image. severity == 0 must be
# a no-op (identity), which the tests enforce.
# --------------------------------------------------------------------------- #
def _defocus_blur(img: Image.Image, severity: float, rng: np.random.Generator) -> Image.Image:
    if severity <= 0:
        return img
    radius = severity * 6.0
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _motion_blur(img: Image.Image, severity: float, rng: np.random.Generator) -> Image.Image:
    """Approximate directional motion blur by averaging integer-pixel shifts
    along a random angle. Avoids a scipy/cv2 dependency; good enough to move
    the same needle (loss of high-frequency detail along one axis) that a
    real motion-blurred re-photograph shows.
    """
    if severity <= 0:
        return img
    length = max(1, int(round(severity * 18)))
    angle = rng.uniform(0, 2 * np.pi)
    dx, dy = np.cos(angle), np.sin(angle)
    arr = np.asarray(img, dtype=np.float32)
    acc = np.zeros_like(arr)
    steps = length
    for i in range(steps):
        t = i - (steps - 1) / 2.0
        sx, sy = int(round(dx * t)), int(round(dy * t))
        acc += np.roll(np.roll(arr, sx, axis=1), sy, axis=0)
    acc /= steps
    return Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8), mode="L")


def _glare(img: Image.Image, severity: float, rng: np.random.Generator) -> Image.Image:
    """A bright Gaussian blob, like a lightbox reflection caught by the phone camera."""
    if severity <= 0:
        return img
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]
    cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
    radius = rng.uniform(0.15, 0.35) * min(h, w) * (0.5 + severity)
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius ** 2))
    arr = arr + blob * 255.0 * severity * rng.uniform(0.6, 1.0)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")


def _shadow(img: Image.Image, severity: float, rng: np.random.Generator) -> Image.Image:
    """A smooth directional gradient that darkens part of the frame."""
    if severity <= 0:
        return img
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]
    angle = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w]
    grad = xx * np.cos(angle) + yy * np.sin(angle)
    span = grad.max() - grad.min()
    grad = (grad - grad.min()) / (span if span > 1e-6 else 1.0)
    strength = severity * rng.uniform(0.3, 0.65)
    arr = arr * (1.0 - grad * strength)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")


def _rotation(img: Image.Image, severity: float, rng: np.random.Generator) -> Image.Image:
    """A handheld off-angle shot, not a perfectly flat re-photograph."""
    if severity <= 0:
        return img
    max_degrees = 12.0
    angle = rng.uniform(-1.0, 1.0) * severity * max_degrees
    fill = int(np.asarray(img, dtype=np.float32).mean())
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=fill)


def _compression(img: Image.Image, severity: float, rng: np.random.Generator) -> Image.Image:
    """JPEG re-encoding artifacts, as if the photo were shared over messaging."""
    if severity <= 0:
        return img
    quality = max(5, int(round(95 - severity * 80)))
    buf = io.BytesIO()
    img.convert("L").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("L")


def _resolution(img: Image.Image, severity: float, rng: np.random.Generator) -> Image.Image:
    """Downsample then upsample back, simulating a low-resolution phone capture
    of a large film rather than a native high-resolution export.
    """
    if severity <= 0:
        return img
    w, h = img.size
    scale = max(0.08, 1.0 - severity * 0.85)
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), resample=Image.BILINEAR)
    return small.resize((w, h), resample=Image.BILINEAR)


def _noise(img: Image.Image, severity: float, rng: np.random.Generator) -> Image.Image:
    """Sensor noise, heavier in the low-light lightbox-photo regime."""
    if severity <= 0:
        return img
    arr = np.asarray(img, dtype=np.float32)
    sigma = severity * 22.0
    noisy = arr + rng.normal(0.0, sigma, arr.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), mode="L")


_KIND_FUNCS = {
    "defocus_blur": _defocus_blur,
    "motion_blur": _motion_blur,
    "glare": _glare,
    "shadow": _shadow,
    "rotation": _rotation,
    "compression": _compression,
    "resolution": _resolution,
    "noise": _noise,
}

DEGRADATION_KINDS = list(_KIND_FUNCS.keys())

# The "simple blur+noise" baseline other degradation-simulation papers use,
# for the strategy-comparison ablation below.
SIMPLE_KINDS = ["defocus_blur", "noise"]

# strategy name -> ordered list of kinds to compose, or None if not implemented
# yet (calling it raises NotImplementedError explaining the missing asset).
DEGRADATION_STRATEGIES: dict[str, list[str] | None] = {
    "simple": SIMPLE_KINDS,
    "full": DEGRADATION_KINDS,
    # Needs a phone-camera degradation model (trained on paired clean/phone-photo
    # data we do not have) rather than hand-specified kernels.
    "gan": None,
    # Needs actual printed films re-photographed with a phone, to compare
    # synthetic severity against ground truth. See docs/DEGRADATION.md.
    "rephoto": None,
}


def apply_degradation(
    img: Image.Image, kind: str, severity: float, rng: np.random.Generator | None = None
) -> Image.Image:
    """Apply a single named degradation at `severity` in [0, 1]. severity <= 0 is a no-op."""
    if kind not in _KIND_FUNCS:
        raise ValueError(f"unknown degradation kind {kind!r}, choose from {DEGRADATION_KINDS}")
    rng = rng or np.random.default_rng()
    severity = float(np.clip(severity, 0.0, 1.0))
    if img.mode != "L":
        img = img.convert("L")
    return _KIND_FUNCS[kind](img, severity, rng)


def compose_degradation(
    img: Image.Image,
    severity: float,
    rng: np.random.Generator | None = None,
    strategy: str = "full",
) -> tuple[Image.Image, dict[str, float]]:
    """Apply a composed smartphone-capture degradation at overall `severity`.

    Returns (degraded_image, applied) where `applied` maps kind -> the actual
    per-kind severity used (each kind is jittered around the target so a
    composed image is not identically-scaled in every channel). Record
    `applied` alongside the image so downstream analysis knows exactly what
    happened, not just the nominal severity.
    """
    if strategy not in DEGRADATION_STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, choose from {sorted(DEGRADATION_STRATEGIES)}")
    kinds = DEGRADATION_STRATEGIES[strategy]
    if kinds is None:
        raise NotImplementedError(
            f"strategy {strategy!r} is a placeholder for a comparison ablation, not "
            "implemented yet. See docs/DEGRADATION.md for what it needs."
        )

    rng = rng or np.random.default_rng()
    severity = float(np.clip(severity, 0.0, 1.0))
    if img.mode != "L":
        img = img.convert("L")
    if severity == 0.0:
        return img, {k: 0.0 for k in kinds}

    applied: dict[str, float] = {}
    out = img
    for kind in kinds:
        k_severity = float(np.clip(severity * rng.uniform(0.6, 1.0), 0.0, 1.0))
        out = apply_degradation(out, kind, k_severity, rng)
        applied[kind] = round(k_severity, 4)
    return out, applied


def severity_to_target_uncertainty(severity):
    """Map degradation severity in [0, 1] to the weak-supervision label the
    roadmap describes: "heavy degradation -> high uncertainty is appropriate".
    Accepts a scalar or an array-like; returns the same shape.

    Scope note: the project picked MC-dropout + temperature scaling over a
    learned confidence head (see the Phase 1 uncertainty-methods survey), so
    this is no longer a training target. It is exactly the same weak label,
    used instead as a *validation* signal: correlate it against the model's
    actual MC-dropout / ensemble uncertainty in
    `xctb/eval/degradation_uncertainty.py`. If a future iteration adds a
    supervised confidence head, this is the label to train it on.
    """
    clipped = np.clip(np.asarray(severity, dtype=float), 0.0, 1.0)
    return float(clipped) if clipped.ndim == 0 else clipped


def build_degradation_manifest(
    manifest,
    severities=(0.0, 0.25, 0.5, 0.75, 1.0),
    strategy: str = "full",
    seed: int = 0,
):
    """Expand a manifest into one row per (image, severity), tagged with (a)
    the existing clinic/cohort and TB-status columns, and (c) the degradation
    strategy, severity, and a per-row seed.

    Pixels are not touched here and no files are copied: `image_path` still
    points at the original image. `xctb.data.dataset.CXRDataset` applies
    `compose_degradation` on the fly at load time, keyed by
    (degradation_strategy, degradation_severity, degradation_seed), so the
    same row degrades identically every epoch without doubling disk usage
    before any real cohort images exist.
    """
    import pandas as pd

    if strategy not in DEGRADATION_STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, choose from {sorted(DEGRADATION_STRATEGIES)}")

    rng = np.random.default_rng(seed)
    rows = []
    for _, row in manifest.iterrows():
        for severity in severities:
            severity = float(severity)
            rows.append(
                {
                    "image_path": row["image_path"],
                    "label": int(row["label"]),
                    "cohort": row["cohort"],
                    "degradation_strategy": strategy if severity > 0 else "none",
                    "degradation_severity": severity,
                    "degradation_seed": int(rng.integers(0, 2**31 - 1)),
                }
            )
    return pd.DataFrame(rows)
