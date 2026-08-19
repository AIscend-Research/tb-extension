"""A printable phantom film: the thing to point a real phone at.

Everything in the physics track is currently validated against `film.py`, the
forward model in this same package. That is a closed loop: it shows the estimator
recovers the parameters we handed it, which is a necessary check and not an
interesting one. The open loop needs a real phone, a real lens, real veiling
glare, a real sensor and a real JPEG encoder in the path -- and, on the other
side of them, something whose optical density we *know*.

Real clinical films give the first half and not the second: there is no
ground-truth density map for a sheet off a hospital archive, so a real-film
pilot can only measure repeatability and the direction of an effect. Archive
PNGs printed on transparency give neither half, because the archives are cropped
and carry none of the three fiducials the inversion needs -- which is what
`audit_fiducials.py` measured at 5.5% coverage.

So: manufacture the film. This module lays out a sheet that carries

* the three fiducials `fiducials.detect` looks for, at the geometry
  `film.add_fiducials` paints them (collimation border, direct-exposure rim,
  lead marker), so the blind estimator runs on it unmodified;
* a **density staircase**, for the tone curve;
* a **detectability grid** -- discs at a ladder of density contrasts x a ladder
  of diameters, spanning and bracketing the four findings in `findings.py` -- so
  the certificate's central claim can be falsified on real captures rather than
  only in simulation;
* a **slanted edge** at 5 degrees inside the field, giving an MTF measurement
  independent of the collimation edge the estimator uses;
* a **flat field**, for read noise and for the veil;
* a **millimetre scale**, so `px_per_mm` is measured rather than inferred from
  the assumed cassette diagonal.

The one thing this module does NOT claim is the realized density of the print.
A printer's transfer is unknown and its maximum density is nowhere near film's
3.2 OD, so the values here are *targets*. What is actually on the sheet is
measured from a reference capture against a calibrated step wedge in the same
frame -- see `recapture.characterize`. Targets are for laying the sheet out;
truth comes off the wedge.

    from tbtrust.physics import phantom
    ph = phantom.build()                    # layout + target density map
    phantom.print_image(ph)                 # 8-bit sheet to send to the printer
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .density import D_MIN_DEFAULT, FilmModel
from .film import FiducialSpec, add_fiducials

# Cassette the canonical frame represents, short x long, in millimetres. The
# phantom is laid out in these coordinates whatever it is printed at, so a
# "2 mm nodule" stays 2 mm of *film* even when the sheet is printed on A4 at 60%
# scale -- the print scale is recorded and the analysis works in film mm.
CASSETTE_MM = (355.0, 432.0)

# What an inkjet actually reaches on transparency. Film runs to 3.2 OD; a print
# reaching 1.6 is doing well, and a laser printer will not manage 1.2. The rim
# that acts as the optical beam stop therefore cannot be printed at all -- it is
# opaque tape, laid over the printed rim, and the protocol says so. This value
# only sets how the target densities are spread across the sheet.
D_MAX_PRINT = 1.60

# Density contrasts and diameters for the detectability grid. The contrast ladder
# brackets every finding in `findings.core()` with headroom either side, because
# the interesting number is where detection *fails* and a grid that starts above
# the failure point measures nothing. The size ladder stops at 24 mm, which is
# below consolidation's 45 mm: the interior of the sheet is only about 200 mm of
# film wide once the collimation border, the D_max rim, its glare-bleed margin
# and the wedge lane are taken out, and five columns at 24 mm is what fits.
# Nothing is lost that matters -- detection at 24 mm and 0.32 OD is already far
# above any floor this pipeline reports -- but `findings_on_grid` flags the
# extrapolation rather than letting it pass as measured.
# A Stouffer T2115 transmission step wedge: 21 steps, 0.15 OD apart, 1 x 5 inches.
# ~$20, and it is the only absolutely calibrated object in the whole rig -- every
# density this pipeline reports on real captures is referred to it. The lane the
# sheet reserves for it is sized in *paper* millimetres, because the wedge is a
# physical object and does not shrink when the sheet is printed smaller.
WEDGE_STEPS = 21
WEDGE_BASE_OD = 0.05
WEDGE_STEP_OD = 0.15
WEDGE_PAPER_MM = (25.4, 127.0)          # width x length

DELTA_D_LADDER = (0.010, 0.020, 0.040, 0.080, 0.160, 0.320)
SIZE_MM_LADDER = (1.5, 3.0, 6.0, 12.0, 24.0)


@dataclass(frozen=True)
class Region:
    """One addressable area of the sheet, in canonical (row, col) pixels.

    `kind` is what the analysis does with it, not what it looks like: several
    regions render identically and are scored completely differently.
    """

    key: str
    kind: str                      # step | patch | background | edge | flat | scale
    rect: tuple[int, int, int, int]        # y0, x0, y1, x1
    target_d: float
    meta: dict = field(default_factory=dict)

    @property
    def slice(self) -> tuple[slice, slice]:
        y0, x0, y1, x1 = self.rect
        return slice(y0, y1), slice(x0, x1)

    def core(self, inset: float = 0.25) -> tuple[slice, slice]:
        """The middle of the region, away from its own edges.

        Every readout uses this rather than the full rect. A patch read to its
        border mixes in the transition the capture PSF smeared across it, which
        biases the mean toward the surround by an amount that grows with blur --
        i.e. exactly with the thing being measured.
        """
        y0, x0, y1, x1 = self.rect
        dy, dx = round((y1 - y0) * inset), round((x1 - x0) * inset)
        return slice(y0 + dy, y1 - dy), slice(x0 + dx, x1 - dx)


@dataclass
class Phantom:
    """A laid-out sheet: its target density map and everything addressable on it."""

    density: np.ndarray                    # target optical density, canonical frame
    regions: list[Region]
    px_per_mm: float                       # canonical pixels per millimetre of *film*
    film: FilmModel
    d_max_print: float
    # Long edge of the sheet as actually printed. The layout is always in film
    # millimetres -- a 2 mm disc is 2 mm of cassette whatever the paper is -- so
    # printing smaller changes only what the printer has to resolve, and the
    # analysis needs no adjustment. Recorded because the print resolution check
    # and the operator instructions both need it.
    print_long_mm: float = CASSETTE_MM[1]
    fiducial_truth: dict = field(default_factory=dict)

    def region(self, key: str) -> Region:
        for r in self.regions:
            if r.key == key:
                return r
        raise KeyError(f"no region {key!r}; have {[r.key for r in self.regions][:8]}...")

    def of_kind(self, kind: str) -> list[Region]:
        return [r for r in self.regions if r.kind == kind]

    def spec(self) -> dict:
        """JSON-serialisable description. This is what ships beside the print."""
        return {
            "canonical_shape": list(self.density.shape),
            "px_per_mm": self.px_per_mm,
            "cassette_mm": list(CASSETTE_MM),
            "d_min": self.film.d_min,
            "d_max_print": self.d_max_print,
            "print_long_mm": self.print_long_mm,
            "print_scale": self.print_long_mm / CASSETTE_MM[1],
            "regions": [
                {"key": r.key, "kind": r.kind, "rect": list(r.rect),
                 "target_d": r.target_d, **({"meta": r.meta} if r.meta else {})}
                for r in self.regions
            ],
        }


def _lerp_density(level: float, d_min: float, d_max: float) -> float:
    """`level` in [0, 1] -> density, 0 = clearest."""
    return float(d_min + (d_max - d_min) * float(np.clip(level, 0.0, 1.0)))


def _disc(shape: tuple[int, int], radius_px: float) -> np.ndarray:
    """Anti-aliased disc, centred. Soft edges matter: a hard-edged disc printed
    at 600 dpi and photographed is a step the printer cannot make anyway, and a
    binary mask would put an aliasing artifact into the one place the analysis
    is measuring contrast."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r = np.hypot(yy - (h - 1) / 2.0, xx - (w - 1) / 2.0)
    return np.clip(radius_px + 0.5 - r, 0.0, 1.0)


def build(
    size: int = 2048,
    film: FilmModel | None = None,
    spec: FiducialSpec | None = None,
    d_max_print: float = D_MAX_PRINT,
    print_long_mm: float = CASSETTE_MM[1],
    wedge_paper_mm: tuple[float, float] = WEDGE_PAPER_MM,
    wedge_steps: int = WEDGE_STEPS,
    delta_d_ladder: tuple[float, ...] = DELTA_D_LADDER,
    size_mm_ladder: tuple[float, ...] = SIZE_MM_LADDER,
    background_level: float = 0.55,
) -> Phantom:
    """Lay out the sheet. `size` is the canonical long-side resolution in pixels.

    2048 is the default because the smallest patch on the grid (1.5 mm at 4.7
    px/mm) is then 7 px across, which survives a print at 600 dpi and a phone
    photographing the sheet at typical framing. Below about 1200 the small end of
    the size ladder stops being a measurement of the capture and becomes a
    measurement of the layout.
    """
    film = film or FilmModel()
    spec = spec or FiducialSpec()
    long_px = int(size)
    short_px = round(long_px * CASSETTE_MM[0] / CASSETTE_MM[1])
    px_per_mm = long_px / CASSETTE_MM[1]

    # Start at the lung-field background level the detectability grid sits on.
    d = np.full((long_px, short_px), _lerp_density(background_level, film.d_min, d_max_print))

    regions: list[Region] = []
    margin = round(spec.collimation_margin * short_px)
    rim = round(spec.direct_exposure_width * short_px)
    # Usable interior: inside the collimation border and clear of the D_max rim,
    # with one rim-width of slack so nothing the analysis reads is close enough
    # to the optically black rim for its glare bleed to matter.
    y0, y1 = margin + 2 * rim, long_px - margin - 2 * rim
    x0, x1 = margin + 2 * rim, short_px - margin - 2 * rim
    iw, ih = x1 - x0, y1 - y0

    # ---------------------------------------------------------------- staircase
    # Eleven steps clear across the printable range. The estimator's tone curve
    # is anchored on two points (the marker and the rim); this measures whether
    # what it does *between* them is right, which two anchors cannot.
    n_steps = 11
    step_h = round(0.065 * ih)
    for i in range(n_steps):
        sx0 = x0 + round(i * iw / n_steps)
        sx1 = x0 + round((i + 1) * iw / n_steps)
        level = i / (n_steps - 1)
        td = _lerp_density(level, film.d_min, d_max_print)
        d[y0:y0 + step_h, sx0:sx1] = td
        regions.append(Region(f"step_{i:02d}", "step", (y0, sx0, y0 + step_h, sx1), td,
                              {"level": level, "index": i}))

    # ------------------------------------------------------------- flat + edge
    band_y0 = y0 + step_h + round(0.03 * ih)
    band_h = round(0.075 * ih)
    flat_d = _lerp_density(background_level, film.d_min, d_max_print)
    d[band_y0:band_y0 + band_h, x0:x0 + iw // 2] = flat_d
    regions.append(Region("flat", "flat", (band_y0, x0, band_y0 + band_h, x0 + iw // 2), flat_d))

    # Slanted edge at 5 degrees: a straight density step, deliberately not axis
    # aligned, so the ESF can be super-sampled across it. `psf.py` measures the
    # capture PSF off the collimation border; this gives a second, independent
    # measurement inside the field, where the answer actually matters.
    ex0 = x0 + iw // 2 + round(0.02 * iw)
    edge_rect = (band_y0, ex0, band_y0 + band_h, x1)
    ey, ex = np.mgrid[0:band_h, 0:(x1 - ex0)].astype(np.float64)
    tilt = np.tan(np.deg2rad(5.0))
    dark = _lerp_density(0.95, film.d_min, d_max_print)
    light = _lerp_density(0.15, film.d_min, d_max_print)
    frac = np.clip(((x1 - ex0) / 2.0 + tilt * (ey - band_h / 2.0)) - ex + 0.5, 0.0, 1.0)
    d[band_y0:band_y0 + band_h, ex0:x1] = light + (dark - light) * frac
    regions.append(Region("slanted_edge", "edge", edge_rect, float("nan"),
                          {"angle_deg": 5.0, "d_light": light, "d_dark": dark}))

    # ------------------------------------------------------- detectability grid
    # The reason the sheet exists. Each cell is background at `background_level`
    # with one disc of a known density increment and a known diameter in film mm.
    grid_y0 = band_y0 + band_h + round(0.025 * ih)
    grid_y1 = y1 - round(0.045 * ih)

    # ------------------------------------------------------------- wedge lane
    # Reserved *first*, down the right-hand side, so the grid is laid out in what
    # is left rather than being overwritten. Vertical rather than horizontal
    # because the grid is height-limited: taking a strip of width costs the size
    # ladder nothing, and taking a strip of height would push the top rung off
    # the sheet.
    #
    # The lane prints clear, with an outline to align the wedge against. It sits
    # inside the collimation field on purpose: rectification then puts it at a
    # known place in the canonical frame, and reading the reference needs no
    # hand-marked corners on every photograph. The blind estimator sees it too,
    # which is fine -- it is just structure on the film, and the estimator is
    # given no clue that those 21 rectangles have known densities.
    px_per_paper_mm = px_per_mm * CASSETTE_MM[1] / float(print_long_mm)
    lane_w = round(wedge_paper_mm[0] * px_per_paper_mm)
    lane_len = round(wedge_paper_mm[1] * px_per_paper_mm)
    gap = round(0.02 * iw)
    if lane_w + gap >= iw // 2 or lane_len > grid_y1 - grid_y0:
        raise ValueError(
            f"a {wedge_paper_mm[0]}x{wedge_paper_mm[1]} mm wedge does not fit the sheet at "
            f"print_long_mm={print_long_mm} (lane {lane_w}x{lane_len} px in a "
            f"{iw}x{grid_y1 - grid_y0} px interior). Print larger or use a shorter wedge.")
    lane_x1 = x1
    lane_x0 = lane_x1 - lane_w
    lane_y0 = grid_y0
    d[lane_y0:lane_y0 + lane_len, lane_x0:lane_x1] = film.d_min
    outline = max(2, round(0.5 * px_per_paper_mm))
    for edge in (slice(lane_y0, lane_y0 + outline), slice(lane_y0 + lane_len - outline, lane_y0 + lane_len)):
        d[edge, lane_x0:lane_x1] = _lerp_density(1.0, film.d_min, d_max_print)
    d[lane_y0:lane_y0 + lane_len, lane_x0:lane_x0 + outline] = _lerp_density(1.0, film.d_min, d_max_print)
    d[lane_y0:lane_y0 + lane_len, lane_x1 - outline:lane_x1] = _lerp_density(1.0, film.d_min, d_max_print)
    step_len = (lane_len - 2 * outline) / wedge_steps
    for i in range(wedge_steps):
        sy0 = lane_y0 + outline + round(i * step_len)
        sy1 = lane_y0 + outline + round((i + 1) * step_len)
        regions.append(Region(f"wedge_{i:02d}", "wedge", (sy0, lane_x0 + outline, sy1, lane_x1 - outline),
                              float("nan"),
                              {"known_od": WEDGE_BASE_OD + WEDGE_STEP_OD * i, "index": i}))
    regions.append(Region("wedge_lane", "wedge_lane", (lane_y0, lane_x0, lane_y0 + lane_len, lane_x1),
                          film.d_min, {"paper_mm": list(wedge_paper_mm), "steps": wedge_steps}))

    grid_x1 = lane_x0 - gap
    grid_w = grid_x1 - x0
    n_rows, n_cols = len(delta_d_ladder), len(size_mm_ladder)
    cell_h = (grid_y1 - grid_y0) // n_rows
    cell_w = grid_w // n_cols
    bg = _lerp_density(background_level, film.d_min, d_max_print)
    for i, dd in enumerate(delta_d_ladder):
        for j, mm in enumerate(size_mm_ladder):
            cy0 = grid_y0 + i * cell_h
            cx0 = x0 + j * cell_w
            cell = (cy0, cx0, cy0 + cell_h, cx0 + cell_w)
            d[cy0:cy0 + cell_h, cx0:cx0 + cell_w] = bg
            r_px = 0.5 * mm * px_per_mm
            if 2 * r_px + 2 >= min(cell_h, cell_w):
                # The ladder outgrew the cell. Skipping beats clipping -- a
                # truncated disc is a different target with a different template
                # energy, and it would be scored as if it were the round one --
                # but a skip must be visible, so `sanity` counts patches against
                # the ladders and says so rather than leaving a hole in the grid.
                continue
            box = round(2 * r_px) + 2
            by0 = cy0 + (cell_h - box) // 2
            bx0 = cx0 + (cell_w - box) // 2
            alpha = _disc((box, box), r_px)
            # Discs go *darker* than background: on developed film more X-ray
            # exposure means more density, and a lesion that attenuates less
            # (a cavity) is the sign-flipped case the certificate treats
            # identically, since the floor bounds |Delta D|.
            d[by0:by0 + box, bx0:bx0 + box] += dd * alpha
            regions.append(Region(
                f"patch_d{dd:.3f}_s{mm:g}", "patch", (by0, bx0, by0 + box, bx0 + box), bg + dd,
                {"delta_d": float(dd), "size_mm": float(mm), "radius_px": float(r_px),
                 "cell": list(cell)}))
            # The paired background is a strip of the cell *beside* the disc, not
            # the cell itself: reading the whole cell would average the disc back
            # in, and the bias would grow with exactly the contrast being
            # measured -- largest where the measurement matters least.
            strip_w = max(4, (cell_w - box) // 2 - 2)
            regions.append(Region(
                f"bg_d{dd:.3f}_s{mm:g}", "background",
                (cy0 + 2, cx0 + 2, cy0 + cell_h - 2, cx0 + 2 + strip_w), bg,
                {"pairs_with": f"patch_d{dd:.3f}_s{mm:g}",
                 "delta_d": float(dd), "size_mm": float(mm)}))

    # ---------------------------------------------------------------- mm scale
    # A ruler, because `invert.py` gets px_per_mm from the assumed cassette
    # diagonal (CASSETTE_MM in invert.py) and that assumption is wrong by
    # whatever the print scale is. Measuring it is the difference between a floor
    # quoted per millimetre and one quoted per pixel.
    scale_y0 = y1 - round(0.05 * ih)
    scale_d = _lerp_density(0.9, film.d_min, d_max_print)
    d[scale_y0:y1, x0:x1] = _lerp_density(0.1, film.d_min, d_max_print)
    ticks_mm = []
    mm = 0.0
    while x0 + round(mm * px_per_mm) < x1:
        tx = x0 + round(mm * px_per_mm)
        tall = abs(mm % 50.0) < 1e-9
        th = (y1 - scale_y0) if tall else round(0.55 * (y1 - scale_y0))
        tw = max(2, round(0.4 * px_per_mm))
        d[scale_y0:scale_y0 + th, tx:tx + tw] = scale_d
        ticks_mm.append(mm)
        mm += 10.0
    regions.append(Region("mm_scale", "scale", (scale_y0, x0, y1, x1), scale_d,
                          {"tick_spacing_mm": 10.0, "n_ticks": len(ticks_mm),
                           "px_per_mm_true": px_per_mm}))

    # Fiducials last: they overwrite the border unconditionally, so anything
    # placed under them would be silently destroyed rather than warned about.
    d, truth = add_fiducials(d, film=film, spec=spec)
    # The rim and marker are painted at *film* densities (3.2 / 0.2 OD) because
    # that is what the estimator's anchors mean. On the sheet the rim is opaque
    # tape and the marker is bare transparency, so those two are physically
    # right even though the printer cannot reach either.
    if not 0.2 <= print_long_mm / CASSETTE_MM[1] <= 1.5:
        raise ValueError(
            f"print_long_mm={print_long_mm} is {print_long_mm / CASSETTE_MM[1]:.2f}x a real "
            "cassette; outside 0.2-1.5 the print either cannot resolve the small end of the "
            "size ladder or will not fit on any paper you have")
    return Phantom(density=d, regions=regions, px_per_mm=px_per_mm, film=film,
                   d_max_print=d_max_print, print_long_mm=float(print_long_mm),
                   fiducial_truth=truth)


BUILD_KEYS = ("size", "print_long_mm", "d_max_print", "wedge_paper_mm", "wedge_steps",
              "delta_d_ladder", "size_mm_ladder", "background_level")


def build_from(args: dict) -> Phantom:
    """Rebuild the exact sheet from the arguments recorded beside a print.

    The analysis has to reconstruct the layout the paper was printed from, and
    "the defaults" is not good enough: change a ladder rung six months later and
    every stored capture would be read against the wrong rectangles, silently.
    `make_phantom_film.py` writes these next to the print for that reason.
    """
    unknown = set(args) - set(BUILD_KEYS)
    if unknown:
        raise ValueError(f"unknown phantom build keys {sorted(unknown)}; expected {list(BUILD_KEYS)}")
    kw = dict(args)
    for k in ("wedge_paper_mm", "delta_d_ladder", "size_mm_ladder"):
        if k in kw and kw[k] is not None:
            kw[k] = tuple(kw[k])
    return build(**kw)


def with_wedge(ph: Phantom) -> np.ndarray:
    """Density map with the calibrated wedge painted into its lane.

    For simulation only. On the real sheet the lane prints clear and a physical
    wedge is taped into it; this paints its nominal step densities so the whole
    characterise-and-score path can be exercised end to end without a printer,
    a lightbox or a phone. Anything that depends on the wedge being *physically*
    there -- its own base fog, the air gap under it, the fact that it is a
    different material from the transparency -- is exactly what a dry run cannot
    tell you, and is why the protocol still calls for reference captures.
    """
    d = np.array(ph.density, dtype=np.float64, copy=True)
    for r in ph.of_kind("wedge"):
        d[r.slice] = r.meta["known_od"]
    return d


def print_image(ph: Phantom, print_gamma: float = 1.0) -> np.ndarray:
    """Target density map -> the 8-bit sheet to send to the printer.

    The mapping is deliberately naive: ink coverage linear in target density,
    with one optional gamma. A printer's real transfer is unknown, nonlinear and
    device-specific, and pretending otherwise here would put a fake calibration
    into the one artifact that must not carry one. What lands on the transparency
    is *measured* afterwards, off the calibrated wedge, by `recapture.characterize`.

    The rim prints as solid black so it is visible for taping over, and the
    marker prints as bare white so the transparency shows through.
    """
    d = np.asarray(ph.density, dtype=np.float64)
    lo, hi = ph.film.d_min, ph.d_max_print
    cover = np.clip((d - lo) / max(hi - lo, 1e-9), 0.0, 1.0) ** float(print_gamma)
    return np.clip(np.rint(255.0 * (1.0 - cover)), 0, 255).astype(np.uint8)


def findings_on_grid(findings, ph: Phantom) -> list[dict]:
    """Where each `findings.py` entry falls on the detectability ladder.

    Reported by the analysis so a reader can see whether a finding is bracketed
    by the grid or sits off the end of it -- an extrapolated verdict and a
    measured one should not read the same on the page.
    """
    dd = np.asarray([r.meta["delta_d"] for r in ph.of_kind("patch")], dtype=float)
    mm = np.asarray([r.meta["size_mm"] for r in ph.of_kind("patch")], dtype=float)
    # Reported per axis: a finding off the end of the size ladder and one off the
    # end of the contrast ladder fail for different reasons and want different
    # rungs added.
    return [
        {
            "key": f.key,
            "delta_d": f.delta_d,
            "size_mm": f.size_mm,
            "delta_d_bracketed": bool(dd.min() <= f.delta_d <= dd.max()),
            "size_bracketed": bool(mm.min() <= f.size_mm <= mm.max()),
            "nearest_patch": f"patch_d{dd[np.argmin(abs(dd - f.delta_d))]:.3f}"
                             f"_s{mm[np.argmin(abs(mm - f.size_mm))]:g}",
        }
        for f in findings
    ]


def sanity(ph: Phantom) -> dict:
    """Cheap checks that the sheet is printable and readable. Run before printing."""
    patches = ph.of_kind("patch")
    smallest = min((r.meta["size_mm"] for r in patches), default=float("nan"))
    return {
        "canonical_shape": list(ph.density.shape),
        "px_per_mm": round(ph.px_per_mm, 3),
        "n_patches": len(patches),
        "n_patches_expected": len(DELTA_D_LADDER) * len(SIZE_MM_LADDER),
        "patches_skipped_too_big_for_cell":
            len(DELTA_D_LADDER) * len(SIZE_MM_LADDER) - len(patches),
        "n_steps": len(ph.of_kind("step")),
        "smallest_patch_mm": smallest,
        "smallest_patch_px": round(smallest * ph.px_per_mm, 1),
        "print_long_mm": ph.print_long_mm,
        "smallest_patch_mm_on_paper": round(smallest * ph.print_long_mm / CASSETTE_MM[1], 2),
        "density_range": [round(float(ph.density.min()), 3), round(float(ph.density.max()), 3)],
        # The rim and marker sit at film densities; everything printable must not.
        "printable_max_d": round(float(np.max(ph.density[ph.density < ph.film.d_max - 1e-6])), 3),
        "min_d_is_base_fog": bool(abs(float(ph.density.min()) - D_MIN_DEFAULT) < 1e-6),
    }
