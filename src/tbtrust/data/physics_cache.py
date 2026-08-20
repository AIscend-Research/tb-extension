"""Precomputed per-pixel floors, so the physics can reach the training loop.

`eval/physics_deferral.py` uses the certificate as a post-hoc gate: the network
is trained, and only then does the physics decide whether to trust it. The
obvious next question is whether measured capture quality helps the classifier
*learn* rather than only helps it abstain -- as a fourth input channel, or as a
per-sample loss weight.

Two facts about this codebase make that harder than it sounds, and both are the
reason this module exists rather than a few lines in `TBDataset`.

**1. The training degradation and the physics capture model are different
pipelines, and only one of them can be certified.** `data/degradation.py` applies
blur, glare, shadow, angle, JPEG and downscale directly to an archive scan. There
is no lead L/R marker in that image, no direct-exposure region and no collimation
border -- so `physics/invert.py` has nothing to calibrate against and the
certificate abstains on *every* training image. The physics arms therefore have
to train on `physics/film.simulate` captures, which lay the fiducials on the film
before photographing it. That is not a detail of convenience: it means the
comparison to the baseline is only fair if the baseline trains on the same
captures, which is what `scripts/measure_physics_in_training.py` enforces.

**2. The floor costs about a second per image.** Inversion runs a tone fit, a
slanted-edge PSF estimate and a glare surface fit, then an FFT per finding.
Recomputing that inside `__getitem__` would make an epoch take longer than the
whole experiment. So it is precomputed once per (image, severity) and cached,
which fixes severity to a discrete grid -- the training augmentation loses its
continuous severity draw in exchange for a physics channel that is real. The
comparison arms all inherit the same discretisation, so it costs the contrast
nothing, but it does mean these runs are not directly comparable to numbers from
`configs/loco_*.yaml`.

Layout, one `.npz` per (image, severity):

    photo   uint8  (S, S)     the capture, exactly what the classifier sees
    floor   f16    (S, S)     per-pixel minimum resolvable |delta D|, worst finding
    mask    bool   (S, S)     lung field, so the floor can be summarised honestly
    meta    json   scalars    margin_db, abstained, limiting_factor, verdict

Stored at the training resolution rather than at 1024: the floor is a smooth
field (it is built from a veil surface and a PSF, neither of which has fine
structure), so downsampling it costs little, while storing 1024x1024 float maps
for 800 images x 5 severities would be 6 GB of disk to feed a 224 px network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The severity grid. Five points, matching the certificate corpus in
#: `scripts/physics_deferral_real.py` so the two are directly comparable.
SEVERITIES = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Floors above this are clipped before normalisation. The floor is +inf where
#: the template energy vanishes, and an inf in an input channel is a NaN loss
#: one backward pass later.
FLOOR_CLIP = 1.0


def cache_key(path: str, severity: float) -> str:
    stem = Path(path).stem
    return f"{stem}__s{round(severity * 100):03d}"


@dataclass
class PhysicsItem:
    """One cached capture and its floor."""

    photo: np.ndarray                 # uint8 (S, S)
    floor: np.ndarray                 # float32 (S, S), +inf clipped to FLOOR_CLIP
    mask: np.ndarray                  # bool (S, S)
    meta: dict

    @property
    def abstained(self) -> bool:
        return bool(self.meta.get("abstained", True))


def compute_one(path: str, severity: float, size: int, out_size: int,
                seed: int = 0) -> PhysicsItem:
    """Photograph one archive scan through the physics model and floor it.

    The capture seed is `utils.seed.capture_seed(path, severity, seed)`, the same
    CRC32 the certificate corpus uses, so a photograph cached here is bit for bit
    the one `scripts/physics_deferral_real.py` scored. Otherwise the training
    cache and the evaluation corpus would be two different draws of the capture
    noise, and every joint statement about them would be wrong.
    """
    from PIL import Image

    from ..physics.certificate import certify
    from ..physics.film import simulate
    from ..physics.invert import invert
    from ..utils.seed import capture_seed

    img = Image.open(path).convert("L")
    if max(img.size) != size:
        img = img.resize((size, size), Image.BILINEAR)
    photo, _ = simulate(np.asarray(img), severity=float(severity),
                        rng=np.random.default_rng(capture_seed(path, severity, seed)),
                        size=size)
    photo = np.asarray(photo, dtype=np.uint8)

    def _down(a, order_bool=False):
        im = Image.fromarray(a.astype(np.uint8) if order_bool else a)
        return np.asarray(im.resize((out_size, out_size),
                                    Image.NEAREST if order_bool else Image.BILINEAR))

    meta: dict = {"path": path, "severity": float(severity), "size": int(size)}
    try:
        cal = invert(photo)
        cert = certify(cal)
        meta.update({
            "verdict": cert.verdict.value,
            "margin_db": float(cert.margin_db),
            "limiting_factor": cert.limiting,
            "worst_finding": getattr(cert.worst_finding, "name", None),
            "abstained": cert.verdict.value == "abstain",
        })
    except Exception as exc:                       # inversion is allowed to fail
        meta.update({"verdict": "abstain", "margin_db": float("nan"),
                     "limiting_factor": "inversion_failed", "worst_finding": None,
                     "abstained": True, "error": repr(exc)[:200]})
        cal = None

    if cal is None or meta["abstained"]:
        # An abstention is information, not a hole: the certificate could not be
        # computed, and the arms have to be told that explicitly rather than
        # handed a zero that reads as "no degradation". FLOOR_CLIP is the worst
        # value on the scale, which is what an unmeasurable photograph deserves.
        floor = np.full((out_size, out_size), FLOOR_CLIP, dtype=np.float32)
        mask = np.ones((out_size, out_size), dtype=bool)
    else:
        # `certify` already computed the worst finding's floor map. Recomputing it
        # from `density_floor` would cost a second FFT *and* risk the cached
        # channel disagreeing with the certificate the eval path reports, which
        # is the one inconsistency that would make the whole comparison unreadable.
        floor = np.asarray(cert.floor_map, dtype=np.float32)
        floor = np.nan_to_num(floor, nan=FLOOR_CLIP, posinf=FLOOR_CLIP)
        floor = np.clip(floor, 0.0, FLOOR_CLIP)
        m = cal.lung_field_mask()
        floor = _down(floor.astype(np.float32))
        mask = _down(np.asarray(m, dtype=bool), order_bool=True).astype(bool)
        meta["floor_median_lung"] = float(np.median(floor[mask])) if mask.any() else float("nan")

    return PhysicsItem(photo=_down(photo).astype(np.uint8),
                       floor=np.asarray(floor, dtype=np.float32),
                       mask=np.asarray(mask, dtype=bool), meta=meta)


def save(item: PhysicsItem, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, photo=item.photo, floor=item.floor.astype(np.float16),
                        mask=item.mask, meta=json.dumps(item.meta))


def load(out: Path) -> PhysicsItem:
    z = np.load(out, allow_pickle=False)
    return PhysicsItem(photo=z["photo"], floor=z["floor"].astype(np.float32),
                       mask=z["mask"], meta=json.loads(str(z["meta"])))


def normalise_floor(floor: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Floor -> [0, 1], high = less information survived.

    Log scale, because the floor spans orders of magnitude across severities and
    a linear channel would be a flat grey with a few white pixels. `lo`/`hi` come
    from the *training* split only (`CacheStats`), never refitted per image: a
    per-image normalisation would erase exactly the between-image differences the
    channel is supposed to carry, which is the failure mode that would make this
    experiment silently measure nothing.
    """
    f = np.log10(np.clip(np.asarray(floor, dtype=np.float32), 1e-6, FLOOR_CLIP))
    return np.clip((f - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)


@dataclass
class CacheStats:
    """Normalisation constants and the cache's own summary. Fit on train only."""

    log_lo: float
    log_hi: float
    n: int
    abstain_rate: float
    margin_median: float

    def to_dict(self) -> dict:
        return {"log_lo": self.log_lo, "log_hi": self.log_hi, "n": self.n,
                "abstain_rate": self.abstain_rate, "margin_median": self.margin_median}

    @staticmethod
    def from_dict(d: dict) -> CacheStats:
        return CacheStats(d["log_lo"], d["log_hi"], d["n"], d["abstain_rate"],
                          d["margin_median"])


def fit_stats(items) -> CacheStats:
    """Percentile-clipped log range over the training items."""
    vals, abst, margins = [], [], []
    for it in items:
        f = it.floor[it.mask] if it.mask.any() else it.floor.ravel()
        vals.append(np.log10(np.clip(f, 1e-6, FLOOR_CLIP)))
        abst.append(it.abstained)
        margins.append(it.meta.get("margin_db", np.nan))
    v = np.concatenate(vals) if vals else np.array([0.0])
    return CacheStats(
        log_lo=float(np.quantile(v, 0.01)), log_hi=float(np.quantile(v, 0.99)),
        n=len(items), abstain_rate=float(np.mean(abst)) if abst else float("nan"),
        margin_median=float(np.nanmedian(margins)) if margins else float("nan"))
