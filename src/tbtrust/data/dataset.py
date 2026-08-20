"""Torch Dataset over the manifest, with on-the-fly smartphone degradation.

Returns, per item:
    image             : float tensor CxHxW, normalized
    label             : long (0 normal / 1 TB)
    uncertainty_target: float in [0,1]  (weak supervision for the confidence head)
    severity          : float, the degradation strength actually applied
    clinic_idx        : long, provenance id (domain label for CORAL/DANN/IRM/FiLM)

Set `degradation_severity` to a fixed value for a controlled robustness sweep
(Phase 4: accuracy vs. severity), or pass `severity_sampler` to randomize during
training so the model sees a range of image qualities.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .degradation import SmartphoneDegradation
from .manifest import clinic_index, uncertainty_target_from_severity

try:  # torch is optional at import time so the rest of the package stays light
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TBDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str | None = None,
        image_size: int = 224,
        degradation_severity: float = 0.0,
        severity_sampler: Callable[[], float] | None = None,
        grayscale_to_rgb: bool = True,
        normalize: bool = True,
        seed: int | None = None,
    ):
        if torch is None:
            raise ImportError("TBDataset needs torch. `pip install -e .`")
        self.df = manifest if split is None else manifest[manifest["split"] == split].reset_index(drop=True)
        self.image_size = image_size
        self.degradation_severity = degradation_severity
        self.severity_sampler = severity_sampler
        self.grayscale_to_rgb = grayscale_to_rgb
        self.normalize = normalize
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.df)

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation stream for a new training epoch.

        With `seed` set, the per-item RNG is derived from (seed, epoch, index).
        Without the epoch term every image would get the identical degradation in
        every epoch, which defeats the point of randomised augmentation; without
        the seed the stream comes from OS entropy and the run is unreproducible.
        Call this once per epoch (the training loop does) and leave it alone at
        eval time, where a fixed stream is exactly what you want.
        """
        self.epoch = int(epoch)

    def _item_seed(self, i: int) -> int | None:
        """Reproducible per-(epoch, item) seed, or None to stay fully random.

        `SmartphoneDegradation` constructs a fresh Generator(seed) on every call,
        so handing it one dataset-wide seed would give every image an identical
        draw. Mixing in the index (and the epoch) keeps images different from each
        other and from themselves across epochs, while staying reproducible.
        """
        if self.seed is None:
            return None
        return (self.seed * 1_000_003 + self.epoch * 10_007 + i) % (2**32)

    def _load(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("L").resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.asarray(img)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        arr = self._load(row["path"])

        item_seed = self._item_seed(i)
        if self.severity_sampler is not None:
            # Drive the sampler from the same per-item stream when seeded, so the
            # severity does not depend on the order the DataLoader happens to
            # fetch items in -- which varies with shuffling and worker count, and
            # would otherwise make a seeded run still irreproducible.
            severity = (
                self.severity_sampler(np.random.default_rng(item_seed))
                if item_seed is not None
                else self.severity_sampler()
            )
        else:
            severity = self.degradation_severity

        if severity > 0:
            arr, _ = SmartphoneDegradation(severity=severity, seed=item_seed)(arr)

        arr = arr.astype(np.float32) / 255.0
        if self.grayscale_to_rgb:
            arr = np.stack([arr, arr, arr], axis=0)          # 3xHxW
        else:
            arr = arr[None, ...]                             # 1xHxW
        tensor = torch.from_numpy(arr)

        if self.normalize and self.grayscale_to_rgb:
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            tensor = (tensor - mean) / std

        u_target = uncertainty_target_from_severity(severity)
        return {
            "image": tensor,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "uncertainty_target": torch.tensor(u_target, dtype=torch.float32),
            "severity": torch.tensor(float(severity), dtype=torch.float32),
            # Domain label for the DG objectives (CORAL/DANN/IRM) and the
            # clinic-conditional FiLM embedding. Always emitted -- it costs one
            # int per image and means switching dg.method needs no data-side
            # change. Ignored entirely when dg.method is 'none'.
            "clinic_idx": torch.tensor(clinic_index(row["clinic"]), dtype=torch.long),
        }


@dataclass
class constant_severity:
    """Callable, not a closure/lambda: DataLoader with num_workers > 0 pickles the
    severity_sampler to hand it to worker processes. A `lambda: value` closure
    fails that pickling under the spawn/forkserver start method (macOS/Windows
    always, and POSIX too as of Python 3.14) -- it looks fine in local testing
    on Linux only because fork doesn't need to pickle anything, then breaks the
    first time someone trains on a Mac or a newer Python. A dataclass with
    __call__ pickles fine everywhere."""

    value: float

    def __call__(self) -> float:
        return self.value


@dataclass
class uniform_severity:
    """Same picklability fix as `constant_severity`, for the randomized-severity
    training sampler. Owns its own `np.random.Generator` instead of closing over
    one, for the same reason.

    Accepts an optional `rng`: when `TBDataset` is seeded it passes a per-item
    generator so the severity is a function of (seed, epoch, index) rather than of
    how many times this sampler happens to have been called. The instance's own
    generator advances in DataLoader fetch order, which changes with shuffling and
    with `num_workers` (each worker gets its own copy of the sampler and replays
    the same sequence), so relying on it alone leaves a "seeded" run irreproducible.
    """

    low: float = 0.0
    high: float = 1.0
    seed: int | None = None

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, rng: np.random.Generator | None = None) -> float:
        return float((rng or self._rng).uniform(self.low, self.high))


class PhysicsCachedDataset(Dataset):
    """Serves precomputed physics captures, and the arm's view of the floor.

    One dataset, four arms, chosen by `physics_mode`:

        none        the photograph alone. This is the *control*, and it trains on
                    the identical cached captures, so a difference against it is
                    attributable to the physics rather than to the capture model.
        channel     the normalised floor map appended as a fourth input channel.
        scramble    a fourth channel carrying some *other* image's floor map.
                    The single most important arm here. A floor map is a smooth,
                    low-frequency field correlated with nothing in particular; an
                    extra channel of that kind can act as a regulariser and buy
                    accuracy while carrying no information about the image it is
                    attached to. If `channel` beats `none` and `scramble` beats
                    `none` by the same amount, the physics contributed nothing
                    and the result is about having a fourth channel at all.
        severity    a fourth channel of one constant, the applied severity. Tests
                    the other direction: does the *per-pixel measured* floor beat
                    a single scalar the simulator already knew? If not, the
                    expensive part of the physics is not what is paying.

    `loss_weight` is independent of all four and can be combined with any of
    them; it comes out in the batch and the training loop applies it.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        cache_dir: str | Path,
        split: str | None = None,
        severities=None,
        physics_mode: str = "none",
        stats=None,
        seed: int | None = 0,
        epoch_severity: bool = True,
        loss_weight: str = "none",
        weight_floor: float = 0.25,
    ):
        if torch is None:
            raise ImportError("PhysicsCachedDataset needs torch. `pip install -e .`")
        from .physics_cache import SEVERITIES

        self.df = manifest if split is None else manifest[manifest["split"] == split].reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        # Default to the grid the cache was actually built on, not the module
        # constant: a cache built with --severities 0,1 and a dataset defaulting
        # to five points asks for files that were never written, and the failure
        # lands mid-epoch rather than at construction.
        if severities is None:
            index = Path(cache_dir) / "index.json"
            if index.exists():
                import json as _json

                severities = _json.loads(index.read_text()).get("severities", SEVERITIES)
        self.severities = tuple(SEVERITIES if severities is None else severities)
        self.physics_mode = str(physics_mode).lower()
        self.stats = stats
        self.seed = seed
        self.epoch = 0
        self.epoch_severity = bool(epoch_severity)
        self.loss_weight = str(loss_weight).lower()
        self.weight_floor = float(weight_floor)
        if self.physics_mode not in {"none", "channel", "scramble", "severity"}:
            raise ValueError(f"unknown physics_mode {physics_mode!r}")

    def __len__(self) -> int:
        return len(self.df)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def in_channels(self) -> int:
        return 3 if self.physics_mode == "none" else 4

    def _rng(self, i: int):
        base = 0 if self.seed is None else self.seed
        return np.random.default_rng((base * 1_000_003 + self.epoch * 10_007 + i) % (2**32))

    def _severity(self, i: int) -> float:
        """Draw from the cached grid, not the continuous range.

        The physics is only available where it was precomputed, so severity is
        discrete here where `TBDataset` samples it continuously. Every arm --
        including the no-physics control -- inherits the same grid, so it costs
        the contrast nothing; it does mean these runs are not comparable to
        numbers from `configs/loco_*.yaml`, which is stated in the results.
        """
        if not self.epoch_severity:
            return float(self.severities[0])
        return float(self.severities[int(self._rng(i).integers(len(self.severities)))])

    def _load(self, path: str, severity: float):
        from .physics_cache import cache_key, load

        f = self.cache_dir / f"{cache_key(path, severity)}.npz"
        if not f.exists():
            raise FileNotFoundError(
                f"{f} is missing. Run scripts/build_physics_cache.py first; the "
                "arms cannot fall back to an uncached capture without changing "
                "the images the control sees.")
        return load(f)

    def __getitem__(self, i: int):
        from .physics_cache import normalise_floor

        row = self.df.iloc[i]
        severity = self._severity(i)
        item = self._load(str(row["path"]), severity)

        arr = item.photo.astype(np.float32) / 255.0
        chans = [arr, arr, arr]
        if self.physics_mode == "channel":
            chans.append(normalise_floor(item.floor, self.stats.log_lo, self.stats.log_hi))
        elif self.physics_mode == "scramble":
            # Another image's floor, at the same severity: the control keeps the
            # channel's marginal distribution and destroys only its pairing.
            j = int(self._rng(i + 7919).integers(len(self.df)))
            other = self._load(str(self.df.iloc[j]["path"]), severity)
            chans.append(normalise_floor(other.floor, self.stats.log_lo, self.stats.log_hi))
        elif self.physics_mode == "severity":
            chans.append(np.full_like(arr, float(severity)))

        tensor = torch.from_numpy(np.stack(chans, axis=0))
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor[:3] = (tensor[:3] - mean) / std

        out = {
            "image": tensor,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "uncertainty_target": torch.tensor(
                uncertainty_target_from_severity(severity), dtype=torch.float32),
            "severity": torch.tensor(float(severity), dtype=torch.float32),
            "clinic_idx": torch.tensor(clinic_index(row["clinic"]), dtype=torch.long),
            "margin_db": torch.tensor(float(item.meta.get("margin_db", float("nan"))),
                                      dtype=torch.float32),
            "abstained": torch.tensor(bool(item.abstained)),
            "loss_weight": torch.tensor(self._weight(item), dtype=torch.float32),
        }
        return out

    def _weight(self, item) -> float:
        """Per-sample loss weight from the certificate, in the requested direction.

        `down` is the argued-for one: when the certificate says the photograph
        cannot carry the finding, the archive label is still TB or not-TB, but
        nothing in *this image* supports it -- so the gradient teaches the
        network to predict the label from whatever spurious cue is left, which is
        the definition of a shortcut. Down-weighting those samples should reduce
        it. `up` is the hard-example-mining intuition and is included because it
        is equally plausible a priori and only measurement can separate them.

        Weights are floored at `weight_floor` rather than driven to zero: on this
        corpus a large fraction of photographs are INSUFFICIENT for the worst
        finding, and a hard zero would throw away most of the training set, which
        would confound "physics helps" with "less data hurts".
        """
        if self.loss_weight == "none":
            return 1.0
        m = float(item.meta.get("margin_db", float("nan")))
        if item.abstained or not np.isfinite(m):
            q = 0.0                                  # unmeasurable: treat as worst
        else:
            q = float(np.clip((m + 12.0) / 24.0, 0.0, 1.0))   # -12 dB .. +12 dB
        w = q if self.loss_weight == "down" else (1.0 - q)
        return float(self.weight_floor + (1.0 - self.weight_floor) * w)
