"""The two cheap objects on the lightbox, and how to read them off a photograph.

`docs/DEPLOYMENT_CHECKLIST.md` B2 asks a clinic for two things that cost pennies:

* a **step wedge** taped to the lightbox beside the film. The film itself offers
  only two distinct densities -- base+fog and D_max -- and a monotone power law
  has three parameters, so `tone.fit_tone` has to pin gamma to an sRGB prior and
  propagate its width into every density. A third distinct density breaks that
  degeneracy and gamma becomes a fitted number with an error bar off the
  chi-squared curvature. A wedge offers twenty-one of them.
* a **ruler**, or anything of known length, in the frame. Otherwise `px_per_mm`
  comes from the detected collimation field against an assumed cassette diagonal,
  at roughly +/-20%, and that error propagates into every finding's spatial
  frequency and so into the density floor.

This module is the deployment half of those two lines: finding both objects in a
photograph with no side information beyond "there may be one there", and handing
back what the inversion needs. `scripts/measure_fiducial_value.py` is the other
half -- what they actually buy, measured.

Three details are load-bearing and none of them is obvious.

**The wedge must be subtracted from the base-fog anchor.** The film's clear
margin is one of the two anchors the whole density scale hangs from, and
`fiducials.detect` takes it as everything outside the collimation border. A wedge
taped in that region is *inside that mask*, so left alone it drags the D_min
anchor toward the wedge's own densities -- a systematic error in the one
measurement that was supposed to be exact. `Aids.exclusion_mask` exists for that,
and `invert` applies it before extracting anchors.

**The wedge's own densities are known; its illumination is not.** It sits on the
lightbox next to the film, so it sees a different part of the illumination field
and a different amount of veil than the film does. Both are estimated where it
sits, exactly as for the film's own anchors, or the wedge would import the
lightbox gradient into gamma.

**A ruler is read by tick spacing, not by its length.** Photographing the ends of
a ruler exactly is hard and cropping one is easy; the tick *pitch* survives both,
and a median over many intervals is robust to a few missed ticks in a way that an
endpoint measurement is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _ops
from .density import FilmModel
from .tone import Anchor

# A Stouffer T2115 transmission step wedge: 21 steps, 0.15 OD apart on a base of
# about 0.05, one inch by five. About 20 pounds, and the only absolutely
# calibrated object anywhere in this pipeline. `phantom.py` reserves a lane for
# the same part.
WEDGE_STEPS = 21
WEDGE_BASE_OD = 0.05
WEDGE_STEP_OD = 0.15
WEDGE_PAPER_MM = (25.4, 127.0)


def wedge_densities(steps: int = WEDGE_STEPS, base_od: float = WEDGE_BASE_OD,
                    step_od: float = WEDGE_STEP_OD) -> np.ndarray:
    """Nominal optical density of each step, clearest first."""
    return base_od + step_od * np.arange(int(steps), dtype=float)


@dataclass
class Aids:
    """What the two objects contributed to one photograph.

    `anchors` are extra densitometry points for `tone.fit_tone`; `px_per_mm` is a
    measured scale. Either may be absent -- a clinic that taped a wedge but has no
    ruler is the common case, and the inversion degrades to its prior for the half
    that is missing rather than refusing the half that is present.
    """

    anchors: list[Anchor] = field(default_factory=list)
    px_per_mm: float | None = None
    exclusion_mask: np.ndarray | None = None
    # Where the base-fog anchor should be read from once the frame is wide enough
    # to include the lightbox as well as the sheet. See `margin_band`.
    base_anchor_mask: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def has_wedge(self) -> bool:
        return len(self.anchors) > 0

    @property
    def has_ruler(self) -> bool:
        return self.px_per_mm is not None and np.isfinite(self.px_per_mm)

    def refreshed(self, veil, illumination) -> Aids:
        """The same anchors, re-evaluated against updated veil and illumination.

        The inversion is a fixed-point loop: veil and illumination are only known
        after the first pass, and an anchor whose expected luminance was computed
        against zero veil is wrong by exactly the veil sitting on it.
        """
        from dataclasses import replace as _replace

        m = self.exclusion_mask
        return _replace(self, anchors=[
            _replace(a, veil_luminance=_at(veil, m, a.veil_luminance),
                     illumination=_at(illumination, m, a.illumination))
            for a in self.anchors
        ])

    def summary(self) -> dict:
        return {
            "aid_wedge_steps": len(self.anchors),
            "aid_px_per_mm": float(self.px_per_mm) if self.has_ruler else float("nan"),
            **{f"aid_{k}": v for k, v in self.diagnostics.items()},
        }


def _outside_region(fid, shape: tuple[int, int]) -> np.ndarray:
    """Where an object taped to the lightbox can be: outside the collimated field.

    Falls back to a border band when no field quad was found. That is the case
    where the aids matter most -- a photograph too tight to grade is also one
    whose scale is least certain -- so refusing to look would be exactly backwards.
    """
    if fid is not None and getattr(fid, "field_mask", None) is not None:
        return ~np.asarray(fid.field_mask, dtype=bool)
    h, w = shape
    band = np.ones((h, w), dtype=bool)
    band[int(0.12 * h):int(0.88 * h), int(0.12 * w):int(0.88 * w)] = False
    return band


def margin_band(fid, shape: tuple[int, int], width_frac: float = 0.05) -> np.ndarray | None:
    """The ring of clear film just outside the collimated field.

    `fiducials.detect` takes the base-fog anchor from everything outside the
    field, which is right when the photograph stops at the edge of the sheet and
    wrong the moment it does not. Put a wedge and a ruler on the lightbox and the
    operator will frame wider to include them -- so the anchor region now contains
    bare lightbox, which transmits everything, against film base+fog at 0.2 OD
    which transmits 63%. The median of that mixture is brighter than base+fog by
    an amount that depends on how much lightbox got in frame, and the entire
    density scale hangs from it.

    Geometry fixes it without a threshold: the sheet's unexposed margin is, by
    construction, the band immediately outside the collimation border. Anything
    further out is not film. Returns None when there is no field quad to measure
    the band from, in which case the caller should keep the original mask -- a
    wide frame is still better than no anchor.
    """
    if fid is None or getattr(fid, "field_mask", None) is None:
        return None
    field = np.asarray(fid.field_mask, dtype=bool)
    if not field.any():
        return None
    h, w = shape
    k = max(2, round(width_frac * min(h, w)))
    return _ops.binary_dilate(field, k) & ~field


def _candidate_strips(img: np.ndarray, outside: np.ndarray, min_area: int = 200,
                      min_elongation: float = 2.5, min_fill: float = 0.35):
    """Elongated solid components of the outside region, with their profiles.

    Both aids are strips lying on the lightbox, and both are found the same way;
    what separates them is what their profile does along the strip. The wedge's
    falls monotonically, the ruler's is periodic. Enumerating once and classifying
    twice is not just tidier -- collapsing the whole outside region onto one axis
    to look for ticks, which is the obvious way to find a ruler, averages the
    ruler's rows together with the film margin and the wedge and washes the ticks
    out entirely.
    """
    vals = img[outside]
    if vals.size < 200:
        return []
    lo, hi = np.quantile(vals, [0.02, 0.98])
    if hi - lo < 0.05:
        return []
    cand = outside & (img < hi - 0.15 * (hi - lo))
    cand = _ops.binary_erode(_ops.binary_dilate(cand, 1), 1)
    labels, n = _ops.label_components(cand)
    out = []
    for st in _ops.component_stats(labels, n):
        hh, ww = st["height"], st["width"]
        elong = max(hh, ww) / max(min(hh, ww), 1)
        if st["area"] < min_area or elong < min_elongation or st["fill"] < min_fill:
            continue
        y0, x0, y1, x1 = st["bbox"]
        comp = labels == st["label"]
        sub = img[y0:y1 + 1, x0:x1 + 1]
        sm = comp[y0:y1 + 1, x0:x1 + 1]
        along_rows = (y1 - y0) >= (x1 - x0)
        if along_rows:
            prof = np.array([np.median(sub[i][sm[i]]) if sm[i].any() else np.nan
                             for i in range(sub.shape[0])])
        else:
            prof = np.array([np.median(sub[:, j][sm[:, j]]) if sm[:, j].any() else np.nan
                             for j in range(sub.shape[1])])
        prof = prof[np.isfinite(prof)]
        if prof.size < 16:
            continue
        out.append({"mask": comp, "bbox": (int(y0), int(x0), int(y1), int(x1)),
                    "profile": prof, "along_rows": along_rows, "area": st["area"]})
    return out


def _peak_train(prof: np.ndarray, smooth: float = 8.0):
    """Positions of the dark ticks in a detrended profile, and their regularity."""
    p = prof - _ops.gaussian_blur(prof[None, :], smooth)[0]
    amp = float(np.std(p))
    if amp < 1e-3:
        return [], float("inf"), amp
    below = p < -1.0 * amp
    peaks, i = [], 0
    while i < below.size:
        if below[i]:
            j = i
            while j + 1 < below.size and below[j + 1]:
                j += 1
            peaks.append(0.5 * (i + j))
            i = j + 1
        else:
            i += 1
    if len(peaks) < 3:
        return peaks, float("inf"), amp
    gaps = np.diff(np.asarray(peaks, dtype=float))
    pitch = float(np.median(gaps))
    spread = float(np.median(np.abs(gaps - pitch)) / max(pitch, 1e-9))
    return peaks, spread, amp


def detect_wedge(
    photo_norm: np.ndarray,
    fid=None,
    steps: int = WEDGE_STEPS,
    base_od: float = WEDGE_BASE_OD,
    step_od: float = WEDGE_STEP_OD,
    min_steps: int = 5,
) -> tuple[list[dict], np.ndarray | None, dict]:
    """Find a step wedge outside the collimated field.

    A wedge is the one thing on a lightbox that is a *monotone staircase*: a
    compact strip whose luminance falls in near-equal logarithmic steps along its
    long axis. That signature is what is searched for, rather than a template
    match, because the strip's orientation, length in pixels and how much of it
    the operator got in frame are all unknown.

    Returns the per-step readings, a mask of the strip, and diagnostics. Steps
    that have clipped -- black-crushed at the dense end, blown at the clear end --
    are dropped: they read the same value as their neighbour, and keeping them
    would flatten the transfer exactly where it is steepest.
    """
    img = np.asarray(photo_norm, dtype=np.float64)
    h, w = img.shape
    outside = _outside_region(fid, (h, w))
    diag: dict = {"searched_px": int(outside.sum())}
    if outside.sum() < 200:
        return [], None, {**diag, "reason": "nothing outside the field to search"}

    def _monotonicity(prof):
        """Spearman rank correlation between position and value.

        This is the test that separates a wedge from everything else on a
        lightbox. A ruler is longer, larger and more elongated than a wedge, so
        ranking candidates by size picks the ruler and then reads 21 identical
        "steps" off it -- anchors that all sit at the same pixel value and drag
        gamma to wherever the fit grid ends. A wedge is the only strip whose value
        falls monotonically along its length.
        """
        if prof.size < 8:
            return 0.0
        x = np.arange(prof.size, dtype=float)
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(prof)).astype(float)
        if np.std(rx) < 1e-9 or np.std(ry) < 1e-9:
            return 0.0
        return abs(float(np.corrcoef(rx, ry)[0, 1]))

    best, rejected = None, []
    for c in _candidate_strips(img, outside):
        prof = c["profile"]
        if prof.size < 3 * min_steps:
            rejected.append({"bbox": c["bbox"], "why": f"only {prof.size} px along its axis"})
            continue
        mono, span = _monotonicity(prof), float(np.ptp(prof))
        if mono < 0.85 or span < 0.10:
            rejected.append({"bbox": c["bbox"], "monotonicity": round(mono, 3),
                             "span": round(span, 3), "why": "not a monotone staircase"})
            continue
        score = mono * min(span, 1.0)
        if best is None or score > best[0]:
            best = (score, c, mono, span)
    if best is None:
        return [], None, {**diag, "reason": "no monotone strip outside the field",
                          "rejected": rejected[:5]}

    _, c, mono, span = best
    y0, x0, y1, x1 = c["bbox"]
    strip_mask, prof, along_rows = c["mask"], c["profile"], c["along_rows"]
    diag.update({"monotonicity": round(mono, 3), "span": round(span, 3)})

    # Slice the profile into `steps` equal bands and read each one's middle. The
    # wedge's own step boundaries are not detected: their positions are known by
    # construction once the strip's extent is, and finding them from the data
    # would be a harder problem with no benefit.
    edges = np.linspace(0, prof.size, steps + 1)
    readings = []
    for i in range(steps):
        a, b = int(edges[i]), int(edges[i + 1])
        seg = prof[a:b]
        if seg.size < 3:
            continue
        core = seg[max(1, seg.size // 4):max(2, 3 * seg.size // 4)]
        readings.append({"index": i, "pixel": float(np.median(core)),
                         "pixel_sigma": float(max(_ops.robust_std(core) / np.sqrt(core.size), 1e-4)),
                         "n_px": int(core.size), "density": float(base_od + step_od * i)})

    if len(readings) < min_steps:
        return [], strip_mask, {**diag, "reason": f"only {len(readings)} steps resolved"}

    # Orientation is unknown: the operator may have taped it either way round. The
    # dense end must be the dark end, so if the profile rises with index, the
    # strip was photographed reversed and the density assignment is flipped.
    px = np.array([r["pixel"] for r in readings])
    if np.polyfit(np.arange(px.size), px, 1)[0] > 0:
        for r, d in zip(readings, wedge_densities(steps, base_od, step_od)[::-1][:len(readings)],
                        strict=False):
            r["density"] = float(d)
        diag["reversed"] = True

    # Drop clipped steps: neighbours that read the same value carry no information
    # about the transfer and would flatten it where it is steepest.
    kept = [readings[0]]
    for r in readings[1:]:
        if abs(r["pixel"] - kept[-1]["pixel"]) > 2.0 / 255.0:
            kept.append(r)
    diag.update({"steps_found": len(readings), "steps_used": len(kept),
                 "clipped_dropped": len(readings) - len(kept),
                 "strip_bbox": [y0, x0, y1, x1],
                 "along": "rows" if along_rows else "cols"})
    if len(kept) < min_steps:
        return [], strip_mask, {**diag, "reason": "wedge is clipped over most of its range"}
    return kept, strip_mask, diag


def detect_ruler(photo_norm: np.ndarray, fid=None, tick_spacing_mm: float = 10.0,
                 min_ticks: int = 4) -> tuple[float | None, np.ndarray | None, dict]:
    """Measure px_per_mm from a ruler's tick pitch, outside the collimated field.

    Pitch rather than length, because photographing both ends of a ruler exactly
    is hard and cropping one is easy, while the pitch survives both. The spacing
    is the *median* interval between ticks, so a few missed or merged ticks move
    the answer by nothing.

    Regularity is what separates a ruler from a row of anything else, and it is
    gated rather than merely reported: a false train of four coincidental dark
    spots would otherwise redefine every millimetre downstream, and a scale error
    is silent -- it changes the floor through the finding's spatial frequency and
    nothing anywhere says so.
    """
    img = np.asarray(photo_norm, dtype=np.float64)
    h, w = img.shape
    outside = _outside_region(fid, (h, w))
    if outside.sum() < 200:
        return None, None, {"reason": "nothing outside the field to search"}

    best, rejected = None, []
    for c in _candidate_strips(img, outside):
        peaks, spread, amp = _peak_train(c["profile"])
        cand = {"bbox": c["bbox"], "n_ticks": len(peaks), "interval_spread": round(spread, 3),
                "amplitude": round(amp, 4)}
        if len(peaks) < min_ticks or spread > 0.15:
            rejected.append(cand)
            continue
        pitch = float(np.median(np.diff(np.asarray(peaks, dtype=float))))
        cand["pitch_px"] = pitch
        if best is None or len(peaks) > best["n_ticks"]:
            best = {**cand, "mask": c["mask"]}
    if best is None:
        return None, None, {"reason": "no regular tick train outside the field",
                            "rejected": rejected[:5]}
    mask = best.pop("mask")
    return float(best["pitch_px"] / tick_spacing_mm), mask, best


def read_aids(
    photo_norm: np.ndarray,
    fid=None,
    film: FilmModel | None = None,
    veil: np.ndarray | float = 0.0,
    illumination: np.ndarray | float = 1.0,
    want_wedge: bool = True,
    want_ruler: bool = True,
    tick_spacing_mm: float = 10.0,
    margin_width_frac: float = 0.05,
) -> Aids:
    """Everything the lightbox aids contribute to one photograph.

    `veil` and `illumination` are the current estimates at the point in the
    inversion this is called from. The wedge sits on the lightbox beside the film,
    so it sees a different part of both fields than the film's own anchors do, and
    evaluating them where it actually sits is what stops it importing the lightbox
    gradient straight into gamma.
    """
    film = film or FilmModel()
    img = np.asarray(photo_norm, dtype=np.float64)
    anchors: list[Anchor] = []
    diag: dict = {}
    mask = None

    if want_wedge:
        readings, strip, wdiag = detect_wedge(img, fid)
        diag.update({f"wedge_{k}": v for k, v in wdiag.items() if k != "rejected"})
        mask = strip
        if readings:
            for r in readings:
                m = None
                if strip is not None:
                    m = strip
                anchors.append(Anchor(
                    name=f"wedge_{r['index']:02d}",
                    pixel_mean=r["pixel"],
                    pixel_sigma=r["pixel_sigma"],
                    density=r["density"],
                    n_px=r["n_px"],
                    veil_luminance=_at(veil, m, 0.0),
                    illumination=_at(illumination, m, 1.0),
                ))

    px_per_mm = None
    if want_ruler:
        px_per_mm, _, rdiag = detect_ruler(img, fid, tick_spacing_mm=tick_spacing_mm)
        diag.update({f"ruler_{k}": v for k, v in rdiag.items() if k != "rejected"})

    band = margin_band(fid, img.shape, margin_width_frac)
    if band is not None:
        diag["margin_band_px"] = int(band.sum())
    return Aids(anchors=anchors, px_per_mm=px_per_mm, exclusion_mask=mask,
                base_anchor_mask=band, diagnostics=diag)


def _at(field_arr, mask, default: float) -> float:
    if field_arr is None:
        return float(default)
    if np.ndim(field_arr) == 0:
        return float(field_arr)
    a = np.asarray(field_arr)
    if mask is None or not np.asarray(mask).any():
        return float(np.median(a))
    return float(np.median(a[np.asarray(mask, dtype=bool)]))
