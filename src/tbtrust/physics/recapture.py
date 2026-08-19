"""Score the blind estimator against a real phone, using the printed phantom.

`validate.py` scores `invert.py` against `film.py`: the estimator recovers the
parameters the forward model was given. That is a closed loop and it cannot fail
for the interesting reason. This module closes the open loop -- a real phone, a
real lens, real veiling glare off a real lightbox, a real sensor and a real JPEG
encoder -- by putting something of *known* density on the other side of them.

The instrument is `phantom.py`: a printed sheet carrying the three fiducials the
inversion needs, a detectability grid, a slanted edge, a millimetre scale, and a
lane for a calibrated transmission step wedge. Two stages, and the split is the
whole design:

**Characterise.** A handful of reference captures, taken under the best
conditions the operator can manage, are rectified and the wedge is read. The
wedge's 21 densities are known absolutely, so they give a pixel -> optical
density transfer for that capture, and that transfer turns every other region on
the sheet into a measured density. This is what the print *actually* carries,
which is not what `phantom.build` targeted -- printer transfers are nonlinear and
device-specific, and assuming otherwise would put a fabricated calibration into
the ground truth.

**Score.** Every other capture -- angles, distances, room light, phones -- is
handed to the blind estimator, which knows nothing about wedges or phantoms. Its
recovered density is compared against the characterised truth. The estimator is
never given the wedge readings; the reference path and the blind path share only
the photograph.

What comes out, per capture:

* **tone** -- density bias and RMSE over the staircase, before and after removing
  a constant offset. The split matters: `invert.py` budgets a systematic term it
  admits it cannot pin (the gamma prior, the film's D_min/D_max tolerances) and a
  random term that must survive differencing. An offset-only error is the first;
  anything left after removing it is the second, and only the second invalidates
  the floor.
* **scale** -- `px_per_mm` measured off the printed ruler versus what the
  estimator inferred from the assumed cassette diagonal.
* **blur** -- PSF sigma from the phantom's interior slanted edge versus the
  estimator's, which it takes off the collimation border. Two edges, one answer.
* **veil** -- estimated veil fraction against the capture condition, which is the
  one number the operator manipulated on purpose.
* **detectability** -- for every disc on the grid: is it recoverable by a matched
  filter, and did the certificate say it would be? That 2x2 is the certificate's
  central claim, tested on a real phone instead of on our own forward model.

Nothing here needs the phantom to be perfect. It needs it to be *known*, and the
wedge is what makes it known.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _ops
from . import floor as FL
from .certificate import certify
from .fiducials import Coverage, detect
from .invert import invert
from .phantom import Phantom, with_wedge
from .validate import _matched_filter_weights


def simulate_capture(ph: Phantom, severity: float = 0.3, rng=None, params=None,
                     with_calibrated_wedge: bool = True):
    """Photograph the phantom through the forward model. For dry runs and tests.

    This is emphatically NOT a substitute for the real captures -- it puts
    `film.py` back on both sides of the comparison, which is the closed loop this
    whole module exists to open. It is here so the analysis path can be exercised,
    and its thresholds sanity-checked, before anyone tapes a wedge to a lightbox.
    """
    from .film import capture, sample_params

    rng = rng or np.random.default_rng(0)
    d = with_wedge(ph) if with_calibrated_wedge else ph.density
    p = params or sample_params(float(severity), rng)
    photo, truth = capture(d, p, fiducial_truth=ph.fiducial_truth, film=ph.film, rng=rng)
    return photo, truth


@dataclass
class Rectified:
    """One photograph mapped back into the phantom's canonical frame."""

    image: np.ndarray                      # float in [0, 1], canonical shape
    coverage: Coverage
    homography: np.ndarray | None
    field_quad: np.ndarray | None
    diagnostics: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.homography is not None


def rectify(photo: np.ndarray, ph: Phantom) -> Rectified:
    """Photo -> canonical phantom frame, via the collimation quad.

    The estimator finds that quad anyway (it is what bounds the field), so this
    adds no assumption the inversion does not already make. What it does add is
    the ability to say *which* patch a pixel belongs to, which is the only reason
    the sheet is worth printing.

    Rectification is used for readout only. The blind inversion always runs on
    the untouched photograph, because resampling puts an interpolation blur into
    the image that the PSF estimate would then measure and attribute to the lens.
    """
    img = np.asarray(photo, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=-1)
    if img.max() > 1.5:
        img = img / 255.0

    fid = detect(img)
    if fid.field_quad is None:
        return Rectified(np.zeros(ph.density.shape), fid.coverage, None, None,
                         {"reason": "no collimation quad; the sheet was not fully in frame"})

    y0, x0, y1, x1 = ph.fiducial_truth["field_rect"]
    dst = np.array([[x0, y0], [x1 - 1, y0], [x1 - 1, y1 - 1], [x0, y1 - 1]], dtype=np.float64)
    src = _ops.order_quad(np.asarray(fid.field_quad, dtype=np.float64))
    H = _ops.estimate_homography(src, dst)
    warped = _ops.warp_perspective(img, H, ph.density.shape, fill=np.nan)
    return Rectified(warped, fid.coverage, H, src,
                     {"fill_fraction": float(np.mean(~np.isfinite(warped)))})


def read_regions(rect: Rectified, ph: Phantom, kinds: tuple[str, ...] | None = None) -> dict:
    """Mean and spread of the rectified pixel values inside every region core."""
    out = {}
    if not rect.ok:
        # A failed rectification leaves a zero image, and reading regions out of
        # it would produce a full set of confident-looking numbers describing
        # nothing. Refuse rather than return them.
        return out
    for r in ph.regions:
        if kinds is not None and r.kind not in kinds:
            continue
        v = rect.image[r.core()]
        v = v[np.isfinite(v)]
        if v.size < 4:
            continue
        out[r.key] = {"kind": r.kind, "mean": float(v.mean()), "std": float(v.std()),
                      "n": int(v.size), **r.meta}
    return out


def wedge_transfer(readings: dict, ph: Phantom):
    """Wedge readings -> a pixel-value to optical-density map for that capture.

    Monotone piecewise-linear through the steps, which is the honest amount of
    structure to impose: 21 calibrated points across the range is dense enough
    that interpolating between them beats fitting any two-parameter curve, and a
    parametric fit would smooth over exactly the ISP nonlinearity the wedge is
    there to expose.

    Steps that have clipped -- at either end -- are dropped. A blown highlight
    reads the same value as its neighbour, and keeping it would flatten the
    transfer over a range where it is actually steep.
    """
    steps = [(v["mean"], v["known_od"]) for k, v in readings.items()
             if v["kind"] == "wedge" and np.isfinite(v["mean"])]
    if len(steps) < 6:
        return None, {"reason": f"only {len(steps)} wedge steps read; need >= 6"}
    steps.sort(key=lambda t: t[0])
    px = np.array([s[0] for s in steps], dtype=float)
    od = np.array([s[1] for s in steps], dtype=float)

    keep = np.ones(len(px), dtype=bool)
    keep[1:] &= np.diff(px) > 1e-3                 # clipped / indistinguishable steps
    px, od = px[keep], od[keep]
    if len(px) < 6:
        return None, {"reason": "wedge is clipped over most of its range; re-expose"}

    # Pixel value rises as density falls, so od is decreasing in px; np.interp
    # needs an increasing x, which px already is after the sort.
    def to_od(pixel):
        p = np.asarray(pixel, dtype=float)
        return np.interp(p, px, od, left=od[0], right=od[-1])

    diag = {"n_steps_used": len(px),
            "od_span": float(od.max() - od.min()),
            "px_span": float(px.max() - px.min()),
            # Outside this range the transfer is an extrapolation and the flag
            # says so rather than the number quietly becoming an assumption.
            "px_range": [float(px.min()), float(px.max())],
            "monotone": bool(np.all(np.diff(od) <= 1e-9) or np.all(np.diff(od) >= -1e-9))}
    _ = ph
    return to_od, diag


def characterize(photos, ph: Phantom) -> dict:
    """What the print actually carries, measured off the wedge. Stage one.

    Give it the reference captures -- head-on, diffuse light, exposure locked,
    no glare the operator can see. Returns per-region realized optical density
    with the spread across those captures, which is the number every later
    comparison is against.
    """
    per_region: dict[str, list[float]] = {}
    used, rejected = 0, []
    for i, photo in enumerate(photos):
        rect = rectify(photo, ph)
        if not rect.ok:
            rejected.append({"index": i, **rect.diagnostics})
            continue
        readings = read_regions(rect, ph)
        to_od, diag = wedge_transfer(readings, ph)
        if to_od is None:
            rejected.append({"index": i, **diag})
            continue
        used += 1
        for key, v in readings.items():
            if v["kind"] in ("wedge", "wedge_lane", "edge", "scale"):
                continue
            per_region.setdefault(key, []).append(float(to_od(v["mean"])))

    truth = {}
    for key, vals in per_region.items():
        a = np.asarray(vals, dtype=float)
        truth[key] = {"od": float(a.mean()),
                      # Spread across reference captures, not within one. It is
                      # the reproducibility of the reference itself, and it is
                      # the floor on any agreement the blind estimator can show.
                      "od_sigma": float(a.std(ddof=1)) if a.size > 1 else float("nan"),
                      "n": int(a.size)}
    return {"regions": truth, "n_captures_used": used, "rejected": rejected,
            "reference_reproducibility_od":
                float(np.nanmedian([v["od_sigma"] for v in truth.values()])) if truth else float("nan")}


def _measured_px_per_mm(rect: Rectified, ph: Phantom) -> float:
    """Read the printed ruler: tick spacing in the rectified frame.

    Rectification maps the sheet back onto the canonical layout, so this returns
    the canonical px/mm by construction and is a *check on the rectification*,
    not an independent measurement of the print. It catches the failure that
    matters -- a quad fitted to the wrong edges, which silently rescales every
    millimetre downstream.
    """
    r = ph.region("mm_scale")
    band = rect.image[r.slice]
    if not np.isfinite(band).any():
        return float("nan")
    prof = np.nanmean(band, axis=0)
    if not np.isfinite(prof).all() or prof.size < 32:
        return float("nan")

    # Median spacing between tick peaks, not the dominant Fourier bin. A tick
    # train is a narrow pulse train, so its second harmonic can carry more energy
    # than its fundamental, and the FFT answer then comes out at exactly half the
    # true spacing -- a factor-of-two error that looks entirely plausible.
    prof = prof - prof.mean()
    thr = 0.4 * float(prof.max())
    above = prof > thr
    peaks = []
    i = 0
    while i < above.size:
        if above[i]:
            j = i
            while j + 1 < above.size and above[j + 1]:
                j += 1
            peaks.append(0.5 * (i + j))
            i = j + 1
        else:
            i += 1
    if len(peaks) < 4:
        return float("nan")
    spacing = float(np.median(np.diff(np.asarray(peaks, dtype=float))))
    return float(spacing / r.meta["tick_spacing_mm"])


def _edge_sigma(rect: Rectified, ph: Phantom) -> float:
    """PSF sigma from the phantom's interior slanted edge.

    Independent of the estimator's own edge: `psf.py` measures the collimation
    border, which is at the extreme of the density range and right beside the
    optically black rim. This one sits mid-range in the middle of the field,
    where the findings are.
    """
    r = ph.region("slanted_edge")
    band = rect.image[r.core(0.12)]
    if band.ndim != 2 or band.shape[1] < 24 or not np.isfinite(band).any():
        return float("nan")
    band = np.where(np.isfinite(band), band, np.nanmedian(band))

    # Register the rows on the edge's *measured* position before averaging, not
    # on the 5 degrees the layout nominally has. Rectification carries a residual
    # rotation of a degree or two, and a registration that assumes the design
    # angle leaves the edge drifting across the band -- tens of pixels here, which
    # comes out as blur the lens never produced. Fitting the drift is also the
    # point of tilting the edge at all: the sub-pixel row offsets super-sample the
    # ESF instead of smearing it.
    h, w = band.shape
    x = np.arange(w, dtype=float)
    grad = np.abs(np.gradient(band, axis=1))
    peak = int(np.argmax(grad.mean(axis=0)))
    half = max(8, int(0.15 * w))
    lo, hi = max(peak - half, 0), min(peak + half + 1, w)
    centres, idx = [], []
    for j in range(h):
        g = grad[j, lo:hi]
        if g.sum() <= 0:
            continue
        centres.append(float((g * x[lo:hi]).sum() / g.sum()))
        idx.append(float(j))
    if len(centres) < 8:
        return float("nan")
    slope, intercept = _ops.fit_line_tls(np.asarray(idx), np.asarray(centres))
    fitted = slope * np.asarray(idx) + intercept
    mid = float(np.mean(fitted))
    rows = [np.interp(x + (fitted[k] - mid), x, band[int(j)], left=band[int(j)][0],
                      right=band[int(j)][-1])
            for k, j in enumerate(idx)]
    prof = np.mean(rows, axis=0)

    # Sigma from the 10-90% rise of the ESF, not from its second moment. The
    # moment is the textbook estimator and the wrong one here: veiling glare puts
    # a low, wide halo under the edge, and a second moment weights that tail by
    # the square of its distance, so a 3% halo can triple the answer. The rise
    # distance sees the core only -- which is the quantity `psf.py` reports and
    # therefore the only one this can be compared against. The halo is measured
    # separately, by `glare.py`, off the beam stop.
    lo_v, hi_v = np.quantile(prof, [0.02, 0.98])
    if hi_v - lo_v < 1e-6:
        return float("nan")
    # Restrict to the transition itself before reading the rise off it. Over the
    # whole profile the plateaus are hundreds of samples all sitting at 0 or 1,
    # and interpolating a level through them returns a position from wherever in
    # the plateau the sort happened to put it -- which on a genuinely sharp edge
    # is the far end of the block, and reads as enormous blur.
    gp = np.abs(np.gradient(prof))
    if not np.isfinite(gp).any() or gp.sum() <= 0:
        return float("nan")
    centre = int(np.argmax(gp))
    span = max(6, int(0.12 * prof.size))
    a, b = max(centre - span, 0), min(centre + span + 1, prof.size)
    seg = prof[a:b]
    norm = (seg - lo_v) / (hi_v - lo_v)
    if norm[0] > norm[-1]:
        norm = 1.0 - norm
    x = np.arange(a, b, dtype=float)
    order = np.argsort(norm)
    x10 = float(np.interp(0.10, norm[order], x[order]))
    x90 = float(np.interp(0.90, norm[order], x[order]))
    rise = abs(x90 - x10)
    # 10-90% of a Gaussian edge spread spans 2 * 1.2816 sigma.
    return float(rise / 2.5631)


def _filter_at(dens: np.ndarray, centre: tuple[float, float], size_px: float, psf_sigma: float) -> float:
    """Matched-filter statistic at one site, computed on a local crop.

    `validate._matched_filter_weights` builds the template over the whole frame,
    which is fine for the 320 px simulations it was written for and is not fine
    here: a phantom capture is several megapixels and every disc needs a couple
    of dozen sites, so the full-frame version spends all its time multiplying
    zeros. The weights are identical -- the template is zero outside its own
    neighbourhood by construction -- so cropping changes the cost and not the
    number.
    """
    h, w = dens.shape
    r = int(max(3.0 * size_px, 8.0)) + int(3 * max(psf_sigma, 1.0)) + 2
    cy, cx = float(centre[0]), float(centre[1])
    y0, y1 = int(max(round(cy) - r, 0)), int(min(round(cy) + r + 1, h))
    x0, x1 = int(max(round(cx) - r, 0)), int(min(round(cx) + r + 1, w))
    crop = dens[y0:y1, x0:x1]
    if crop.size == 0:
        return float("nan")
    weights = _matched_filter_weights(crop.shape, (cy - y0, cx - x0), size_px, psf_sigma)
    return float(np.sum(weights * crop))


def detectability(rect: Rectified, cal, ph: Phantom, truth: dict, spec: FL.FloorSpec | None = None) -> list[dict]:
    """Per disc: could a matched filter find it, and did the certificate say so?

    Empirical side: the optimal linear detector for a disc of that size, applied
    at the disc and at signal-free sites in the same cell. d' is the separation
    in units of the background statistic's own spread, so it needs no absolute
    calibration -- which is what lets it be computed on a real capture where the
    absolute density scale is the thing under test.

    Predicted side: the floor `certificate.py` would quote for a finding of that
    size at that site, from the blind inversion alone.

    The two disagree in two directions and they are not symmetric. Predicted
    detectable but empirically invisible is the dangerous one: the certificate
    would have cleared an image that cannot carry the finding.
    """
    spec = spec or FL.FloorSpec()
    rows = []
    dens = cal.density
    px_per_mm = ph.px_per_mm
    for r in ph.of_kind("patch"):
        cell = r.meta["cell"]
        size_px_canonical = 2.0 * r.meta["radius_px"]
        # The recovered density lives in the photo frame; the patch rectangle is
        # canonical. Map the centre through the homography rather than warping
        # the density map, which would blur it.
        cy = 0.5 * (r.rect[0] + r.rect[2])
        cx = 0.5 * (r.rect[1] + r.rect[3])
        if rect.homography is None:
            continue
        Hinv = np.linalg.inv(rect.homography)
        v = Hinv @ np.array([cx, cy, 1.0])
        px, py = float(v[0] / v[2]), float(v[1] / v[2])
        if not (0 <= py < dens.shape[0] and 0 <= px < dens.shape[1]):
            continue
        # Canonical px -> photo px, from the local scale of the same map.
        v2 = Hinv @ np.array([cx + size_px_canonical, cy, 1.0])
        size_px = float(np.hypot(v2[0] / v2[2] - px, v2[1] / v2[2] - py))
        if size_px < 2.0:
            rows.append({"patch": r.key, "skipped": "disc under 2 px in the photograph"})
            continue

        t_sig = _filter_at(dens, (py, px), size_px, float(cal.psf.sigma))

        # Signal-free sites: the same filter, elsewhere in the same cell.
        rng = np.random.default_rng(abs(hash(r.key)) % (2**31))
        stats = []
        for _ in range(24):
            oy = 0.5 * (cell[0] + cell[2]) + rng.uniform(-0.35, 0.35) * (cell[2] - cell[0])
            ox = 0.5 * (cell[1] + cell[3]) + rng.uniform(-0.35, 0.35) * (cell[3] - cell[1])
            # Keep the whole filter footprint off the disc, not just its centre.
            # The template has support out to three diameters, so a site merely
            # outside the disc still integrates part of it -- which lands the
            # signal in the *noise* population and collapses d' exactly where the
            # contrast is highest and the answer should be obvious.
            if np.hypot(oy - cy, ox - cx) < 3.0 * size_px_canonical:
                continue
            vv = Hinv @ np.array([ox, oy, 1.0])
            qx, qy = float(vv[0] / vv[2]), float(vv[1] / vv[2])
            if not (0 <= qy < dens.shape[0] and 0 <= qx < dens.shape[1]):
                continue
            stats.append(_filter_at(dens, (qy, qx), size_px, float(cal.psf.sigma)))
        if len(stats) >= 8:
            arr = np.asarray(stats)
            sd = float(arr.std(ddof=1))
            ref = float(arr.mean())
            noise_model = "empirical_sites"
        else:
            # A disc that nearly fills its cell leaves no room for a signal-free
            # site of the same footprint, so the empirical route runs out exactly
            # at the top of the size ladder. Fall back on the filter's own noise
            # algebra: the weights are unit-norm and zero-mean, so against locally
            # stationary noise of standard deviation sigma the statistic has
            # standard deviation sigma and mean zero. That is a weaker claim -- it
            # assumes the noise is white, which JPEG blocking and the veil-fit
            # residual both violate -- so it is labelled, not silently mixed in.
            bg_key = f"bg_d{r.meta['delta_d']:.3f}_s{r.meta['size_mm']:g}"
            bg_region = next((q for q in ph.regions if q.key == bg_key), None)
            if bg_region is None:
                rows.append({"patch": r.key, "skipped": "no paired background region"})
                continue
            byx = []
            for (qy, qx) in ((bg_region.rect[0], bg_region.rect[1]), (bg_region.rect[2], bg_region.rect[3])):
                vv = Hinv @ np.array([float(qx), float(qy), 1.0])
                byx.append((float(vv[1] / vv[2]), float(vv[0] / vv[2])))
            (ya, xa), (yb, xb) = byx
            y0 = int(np.clip(min(ya, yb), 0, dens.shape[0] - 1))
            y1 = int(np.clip(max(ya, yb), 0, dens.shape[0] - 1))
            x0 = int(np.clip(min(xa, xb), 0, dens.shape[1] - 1))
            x1 = int(np.clip(max(xa, xb), 0, dens.shape[1] - 1))
            patch_bg = dens[y0:y1 + 1, x0:x1 + 1]
            if patch_bg.size < 16:
                rows.append({"patch": r.key, "skipped": "paired background too small to read"})
                continue
            sd = float(_ops.robust_std(patch_bg.ravel()))
            ref = 0.0
            noise_model = "local_variance"
        dprime = float(abs(t_sig - ref) / sd) if sd > 0 else float("inf")

        # Predicted: the floor for a target of this size at this site.
        from .findings import FindingSpec

        f = FindingSpec(key=r.key, name=r.key, delta_d=r.meta["delta_d"], delta_d_sigma=0.0,
                        size_mm=r.meta["size_mm"], size_sigma_mm=0.0, source="PHANTOM")
        fm = FL.density_floor(cal, f, spec)
        yy = int(np.clip(round(py), 0, fm.floor.shape[0] - 1))
        xx = int(np.clip(round(px), 0, fm.floor.shape[1] - 1))
        floor_here = float(fm.floor[yy, xx])

        # Realized contrast, from the characterisation -- not the target, which
        # the printer did not honour exactly.
        t_patch = truth.get(r.key, {}).get("od", float("nan"))
        t_bg = truth.get(r.meta.get("pairs_with", ""), {}).get("od", float("nan"))
        bg_key = f"bg_d{r.meta['delta_d']:.3f}_s{r.meta['size_mm']:g}"
        if not np.isfinite(t_bg):
            t_bg = truth.get(bg_key, {}).get("od", float("nan"))
        realized_dd = float(abs(t_patch - t_bg)) if np.isfinite(t_patch) and np.isfinite(t_bg) else float("nan")

        rows.append({
            "patch": r.key,
            "target_delta_d": r.meta["delta_d"],
            "realized_delta_d": realized_dd,
            "size_mm": r.meta["size_mm"],
            "size_px_photo": round(size_px, 2),
            "dprime": dprime,
            "noise_model": noise_model,
            "empirically_detectable": bool(dprime >= spec.rose_k),
            "predicted_floor": floor_here,
            "predicted_detectable": bool(np.isfinite(realized_dd) and realized_dd >= floor_here),
            "px_per_mm_canonical": px_per_mm,
        })
    return rows


def score_capture(photo: np.ndarray, ph: Phantom, truth: dict,
                  spec: FL.FloorSpec | None = None, meta: dict | None = None) -> dict:
    """One real capture, end to end: blind inversion versus characterised truth."""
    meta = meta or {}
    # `characterize` returns its table under "regions" alongside its own
    # diagnostics; accept either that or a bare region table, because passing the
    # whole thing is the obvious call and silently reading zero regions out of it
    # would look like a capture that scored badly rather than one never scored.
    regions_truth = truth.get("regions", truth) if isinstance(truth, dict) else truth
    rect = rectify(photo, ph)
    if not rect.ok:
        return {"ok": False, **meta, **rect.diagnostics}

    img = np.asarray(photo, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=-1)
    cal = invert(img if img.max() > 1.5 else img * 255.0)
    cert = certify(cal)

    # --- tone: recovered density against the wedge-referenced truth -----------
    Hinv = np.linalg.inv(rect.homography)
    pairs = []
    for r in ph.regions:
        if r.kind not in ("step", "flat", "background"):
            continue
        t = regions_truth.get(r.key, {}).get("od")
        if t is None or not np.isfinite(t):
            continue
        cy = 0.5 * (r.rect[0] + r.rect[2])
        cx = 0.5 * (r.rect[1] + r.rect[3])
        v = Hinv @ np.array([cx, cy, 1.0])
        px, py = float(v[0] / v[2]), float(v[1] / v[2])
        if not (0 <= py < cal.density.shape[0] and 0 <= px < cal.density.shape[1]):
            continue
        half = 3
        y0 = int(np.clip(py - half, 0, cal.density.shape[0] - 1))
        x0 = int(np.clip(px - half, 0, cal.density.shape[1] - 1))
        patch = cal.density[y0:y0 + 2 * half + 1, x0:x0 + 2 * half + 1]
        if patch.size:
            pairs.append((float(np.median(patch)), float(t), r.kind))

    est = np.array([p[0] for p in pairs], dtype=float)
    ref = np.array([p[1] for p in pairs], dtype=float)
    tone: dict = {"n_regions": len(pairs)}
    if len(pairs) >= 4:
        err = est - ref
        offset = float(np.median(err))
        tone.update({
            "bias_od": offset,
            "rmse_od": float(np.sqrt(np.mean(err**2))),
            # After removing the offset: the part that does not cancel in a
            # difference, and therefore the part the floor depends on.
            "rmse_od_after_offset": float(np.sqrt(np.mean((err - offset) ** 2))),
            "slope": float(np.polyfit(ref, est, 1)[0]) if np.std(ref) > 1e-6 else float("nan"),
            "range_od": [float(ref.min()), float(ref.max())],
        })

    measured_px_per_mm = _measured_px_per_mm(rect, ph)
    edge_sigma = _edge_sigma(rect, ph)
    canonical_per_photo_px = float(np.sqrt(abs(np.linalg.det(rect.homography[:2, :2]))))
    return {
        "ok": True,
        **meta,
        "coverage": rect.coverage.value,
        "tone": tone,
        "scale": {
            "canonical_px_per_mm": ph.px_per_mm,
            "rectified_px_per_mm": measured_px_per_mm,
            "estimator_px_per_mm": float(cal.px_per_mm),
            # The estimator gets px/mm from the assumed cassette diagonal. On a
            # phantom printed smaller than a cassette that assumption is wrong by
            # the print scale, and this is the number that says by how much.
            "estimator_over_truth": float(cal.px_per_mm / ph.px_per_mm) if ph.px_per_mm else float("nan"),
        },
        "blur": {
            "estimator_sigma_px_photo": float(cal.psf.sigma),
            # The phantom edge is measured in the rectified frame, the estimator's
            # in the photograph. Comparing the two raw is comparing pixels of
            # different sizes; the homography's linear scale converts one to the
            # other, and without it the disagreement is mostly the framing.
            "estimator_sigma_px_canonical": float(cal.psf.sigma * canonical_per_photo_px),
            "phantom_edge_sigma_px_canonical": edge_sigma,
            "canonical_per_photo_px": canonical_per_photo_px,
            "anisotropy": float(cal.psf.anisotropy),
        },
        "veil": {
            "glare_method": cal.glare.method,
            "veil_sigma": float(cal.glare.veil_sigma),
            "n_probes": int(cal.glare.n_probes),
            **{k: v for k, v in cal.summary().items()
               if k in ("veil_fraction_median", "contrast_retained_median",
                        "sigma_random_median", "sigma_systematic_median",
                        "tone_method", "tone_gamma")},
        },
        "certificate": {"verdict": cert.verdict.value, "margin_db": float(cert.margin_db),
                        "limiting": cert.limiting, "abstained": bool(cert.abstained)},
        "detectability": detectability(rect, cal, ph, regions_truth, spec),
    }


def confusion(rows: list[dict], rose_k: float = 5.0, clear_margin_db: float = 3.0,
              clear_dprime_frac: float = 0.7) -> dict:
    """Collapse the per-disc table into the 2x2 the claim lives or dies on.

    Both directions of disagreement are counted, and one of them is counted
    twice. A disc sitting a hair above the floor and a hair below d' = k is not
    evidence against the bound -- it is the bound being approximately right at
    the point where it is hardest to be right, and both sides of that comparison
    carry noise. So `predicted_detectable_but_invisible` is the raw count, and
    `clear_violations` is the subset where the certificate had real margin
    (>= `clear_margin_db`) and the detector was nowhere close
    (d' <= `clear_dprime_frac` * k). Only the second is a falsification; the
    first is the number to watch drift.
    """
    usable = [r for r in rows if "dprime" in r and np.isfinite(r.get("realized_delta_d", np.nan))]
    tp = sum(1 for r in usable if r["predicted_detectable"] and r["empirically_detectable"])
    tn = sum(1 for r in usable if not r["predicted_detectable"] and not r["empirically_detectable"])
    unsafe = sum(1 for r in usable if r["predicted_detectable"] and not r["empirically_detectable"])
    conservative = sum(1 for r in usable if not r["predicted_detectable"] and r["empirically_detectable"])

    def _margin_db(r):
        fl = r.get("predicted_floor", float("nan"))
        dd = r.get("realized_delta_d", float("nan"))
        if not (np.isfinite(fl) and np.isfinite(dd)) or fl <= 0 or dd <= 0:
            return float("nan")
        return float(20.0 * np.log10(dd / fl))

    clear = [r for r in usable
             if r["predicted_detectable"] and not r["empirically_detectable"]
             and _margin_db(r) >= clear_margin_db and r["dprime"] <= clear_dprime_frac * rose_k]
    n = max(len(usable), 1)
    return {
        "n_discs": len(usable),
        "agree": tp + tn,
        "agreement_rate": (tp + tn) / n,
        "predicted_detectable_but_invisible": unsafe,
        "unsafe_rate": unsafe / n,
        "clear_violations": len(clear),
        "clear_violation_rate": len(clear) / n,
        "clear_violation_examples": [r["patch"] for r in clear[:5]],
        "predicted_insufficient_but_visible": conservative,
        "conservative_rate": conservative / n,
    }
