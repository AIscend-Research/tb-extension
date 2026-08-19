#!/usr/bin/env python3
"""What the two cheap lightbox aids actually buy, measured rather than asserted.

`docs/DEPLOYMENT_CHECKLIST.md` B2 asks a clinic for two things costing pennies:

* a **step wedge** taped beside the film, which should turn the tone curve's
  exponent gamma from an sRGB prior into a fitted parameter with an error bar;
* a **ruler** in the frame, which should settle `px_per_mm` exactly instead of
  inferring it from the detected field against an assumed cassette diagonal.

Both claims are plausible and neither had a number. This script produces them, by
running the same photograph through the inversion four ways -- nothing changes
between arms except which information the estimator is allowed to use:

    baseline   gamma from the sRGB prior, px/mm from the cassette diagonal
    wedge      gamma fitted from the wedge's known densities
    ruler      px/mm measured from the ruler's tick pitch
    both       both aids

and against a fifth, `oracle`, which is handed the true gamma and the true scale.
The oracle is the yardstick: it is what the two aids could buy if they worked
perfectly, so a gap between `both` and `oracle` is the part of the error the aids
do not remove.

Two knobs are swept because both are wrong in the field and neither is under the
estimator's control: the ISP's true gamma (1.8-3.0, i.e. the prior is sometimes
right and sometimes badly wrong) and the collimation tightness, which is what
makes the assumed cassette-diagonal scale wrong by a realistic amount.

    python scripts/measure_fiducial_value.py --out outputs/fiducial_value.json
    python scripts/measure_fiducial_value.py --quick        # a couple of minutes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from tbtrust.physics import _ops
from tbtrust.physics import findings as FIND
from tbtrust.physics.certificate import certify
from tbtrust.physics.density import D_LUNG_RANGE, FilmModel, density_to_transmittance
from tbtrust.physics.film import (
    CaptureParams,
    FiducialSpec,
    add_fiducials,
    add_lightbox_aids,
    capture,
    sample_params,
    synthetic_chest_density,
)
from tbtrust.physics.floor import FloorSpec, density_floor
from tbtrust.physics.invert import CASSETTE_MM, invert
from tbtrust.physics.tone import Anchor, fit_tone

ARMS = ("baseline", "framing", "wedge", "ruler", "both", "oracle")


def fit_isolation(gammas=(1.8, 2.2, 2.6, 3.0), black_levels=(0.0, 0.02)) -> list[dict]:
    """Does `fit_tone` recover gamma at all, given anchors from its own model?

    Run first and reported first, because it separates two very different
    failures. If this passes and the measured arms below still miss gamma, the
    fit is sound and the *anchors* are the problem -- the veil and illumination
    estimated where the wedge sits, the ISP's unmodelled contrast curve. If this
    fails, nothing downstream is worth reading.

    Anchors are generated from v = c0 + c1 * L**(1/gamma), quantised to 8 bits and
    otherwise noiseless: exactly the estimator's model, so the only error left is
    the fit's own.

    Run at two black levels, because the fit pins c0 at zero on purpose (see
    `tone.fit_tone`) and that pinning is not free: against an ISP with a real
    black pedestal the fitted gamma absorbs it and comes out biased. Reporting
    both is the honest way to state the trade -- a slightly biased gamma bought in
    exchange for not letting the tone curve swallow the veil.
    """
    rows = []
    for g in gammas:
      for black in black_levels:
        c0, c1 = float(black), 0.95
        ds = np.concatenate([[0.2, 3.2], 0.05 + 0.15 * np.arange(21)])
        anchors = []
        for d in ds:
            lum = float(density_to_transmittance(float(d)))
            v = float(np.clip(round((c0 + c1 * lum ** (1.0 / g)) * 255) / 255, 0, 1))
            anchors.append(Anchor(f"a{d:.2f}", v, 1e-3, float(d), 500))
        t = fit_tone(anchors)
        rows.append({"gamma_true": g, "black_level": float(black),
                     "gamma_fitted": float(t.gamma),
                     "error": float(t.gamma - g), "sigma": float(t.gamma_sigma),
                     "residual": float(t.residual), "method": t.method})
    return rows


def _quad_mask(quad: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterise the true collimated field, by half-plane intersection."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    inside = np.ones((h, w), dtype=bool)
    c = quad.mean(axis=0)
    for i in range(4):
        (x0, y0), (x1, y1) = quad[i], quad[(i + 1) % 4]
        a, b = (y1 - y0), -(x1 - x0)
        cc = -(a * x0 + b * y0)
        sign = np.sign(a * c[0] + b * c[1] + cc)
        inside &= (sign * (a * xx + b * yy + cc)) >= 0
    return inside


def _cassette_shaped(size: int, rng, film: FilmModel) -> np.ndarray:
    """A film with a real cassette's aspect ratio, not a square.

    `synthetic_chest_density` returns a square, and a square film makes the
    baseline scale error mostly an artefact: the estimator compares the detected
    field diagonal against a 355x432 cassette, so a square sheet is already 20%
    off before collimation is considered. Cropping to the cassette aspect leaves
    collimation tightness and cassette size as the error sources, which are the
    two that vary in a clinic.
    """
    d, _ = synthetic_chest_density(size=size, rng=rng, film=film)
    short = round(size * CASSETTE_MM[0] / CASSETTE_MM[1])
    x0 = (size - short) // 2
    return d[:, x0:x0 + short]


def _floor_summary(cal, findings, spec: FloorSpec) -> dict:
    """Median density floor in the lung field, per finding."""
    m = cal.lung_field_mask()
    out = {}
    for f in findings:
        fm = density_floor(cal, f, spec)
        v = fm.floor[m] if m.any() else fm.floor.ravel()
        out[f.key] = float(np.median(v))
    return out


def run_case(image_seed: int, severity: float, gamma_true: float, collimation: float,
             size: int, film_long_mm: float, film: FilmModel, spec: FloorSpec,
             findings, rng) -> list[dict]:
    """One photograph, inverted five ways."""
    base = _cassette_shaped(size, np.random.default_rng(1000 + image_seed), film)
    base, ftruth = add_fiducials(base, film=film, spec=FiducialSpec(collimation_margin=collimation))
    px_per_mm_true = size / float(film_long_mm)
    padded, aid_truth = add_lightbox_aids(base, film=film, px_per_mm=px_per_mm_true)

    params = sample_params(float(severity), rng)
    params = CaptureParams(**{**params.__dict__, "tone_gamma": float(gamma_true)})
    photo, truth = capture(padded, params, fiducial_truth=ftruth, film=film,
                           rng=np.random.default_rng(int(rng.integers(1 << 31))))

    # Truth in the photo frame, for the density comparison. The padded map is what
    # was photographed, so it is what the homography maps.
    d_true_photo = _ops.warp_perspective(padded, truth.homography, padded.shape, fill=film.d_min)
    # Score inside the *true* collimated field only. The photograph now includes
    # bare lightbox, a wedge and a ruler, and the lung-field heuristic is a
    # heuristic -- letting it wander onto the pad would report the estimator's
    # error on objects it was never asked to measure.
    in_field = _quad_mask(np.asarray(truth.field_quad_photo, dtype=float), padded.shape)
    # One comparison mask for every arm, defined from the *truth*: lung-band
    # density inside the true field. `CalibratedFilm.lung_field_mask` is a
    # heuristic that reads the recovered density, so it moves between arms -- and
    # a metric whose support changes with the arm is comparing different pixels
    # and calling the difference an improvement.
    lung_lo, lung_hi = D_LUNG_RANGE
    score_mask = in_field & (d_true_photo >= lung_lo) & (d_true_photo <= lung_hi)

    rows = []
    for arm in ARMS:
        kw: dict = {}
        if arm == "framing":
            # Aids in frame, neither used: isolates what the wider framing itself
            # costs and what restricting the base anchor to the sheet's margin
            # recovers, so neither is credited to the wedge or the ruler.
            kw["aids"] = ()
        elif arm == "wedge":
            kw["aids"] = ("wedge",)
        elif arm == "ruler":
            kw["aids"] = ("ruler",)
        elif arm == "both":
            kw["aids"] = True
        elif arm == "oracle":
            # Gamma pinned at truth and the scale handed over: what the two aids
            # would buy if each worked perfectly.
            kw["gamma_prior"] = float(gamma_true)
            kw["gamma_sigma"] = 1e-3
            kw["px_per_mm"] = px_per_mm_true

        cal = invert(photo, film=film, **kw)
        cert = certify(cal, findings=findings, spec=spec)
        m = score_mask
        if m.any():
            err = cal.density[m] - d_true_photo[m]
            # Median absolute error, not RMSE. Density error scales as 1/signal,
            # so the densest lung pixels -- where the film passes a percent of the
            # box -- carry an error an order of magnitude above the rest, and an
            # RMSE over the field is a report on those pixels alone. The median
            # describes the field.
            abs_mae = float(np.median(np.abs(err)))
            # The same after removing the common offset: the part that does not
            # cancel in a density *difference*, which is what the floor bounds.
            diff_mae = float(np.median(np.abs(err - np.median(err))))
        else:
            abs_mae = diff_mae = float("nan")

        rows.append({
            "image": image_seed, "severity": severity, "gamma_true": gamma_true,
            "collimation": collimation, "arm": arm,
            "coverage": cal.fiducials.coverage.value,
            "gamma_est": float(cal.tone.gamma),
            "gamma_err": float(cal.tone.gamma - gamma_true),
            "gamma_sigma": float(cal.tone.gamma_sigma),
            "tone_method": cal.tone.method,
            "px_per_mm_true": px_per_mm_true,
            "px_per_mm_est": float(cal.px_per_mm),
            "px_per_mm_rel_err": float((cal.px_per_mm - px_per_mm_true) / px_per_mm_true),
            "density_abs_mae": abs_mae,
            "density_diff_mae": diff_mae,
            "margin_db": float(cert.margin_db),
            "verdict": cert.verdict.value,
            "limiting": cert.limiting,
            **{f"floor_{k}": v for k, v in _floor_summary(cal, findings, spec).items()},
        })
    _ = aid_truth
    return rows


def summarise(rows: list[dict]) -> dict:
    """Per arm, and then the differences that are the point of the exercise."""
    import statistics as st

    def med(vals):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(st.median(vals)) if vals else float("nan")

    per_arm = {}
    for arm in ARMS:
        sel = [r for r in rows if r["arm"] == arm]
        if not sel:
            continue
        gamma_err = [abs(r["gamma_err"]) for r in sel]
        covered = [abs(r["gamma_err"]) <= 2 * r["gamma_sigma"] for r in sel]
        per_arm[arm] = {
            "n": len(sel),
            "abs_gamma_err_median": med(gamma_err),
            "gamma_sigma_median": med([r["gamma_sigma"] for r in sel]),
            # Does the quoted error bar contain the truth as often as it claims?
            # A 2-sigma interval should hold about 95% of the time; anything far
            # below that is an error bar that cannot be quoted.
            "gamma_2sigma_coverage": float(np.mean(covered)) if covered else float("nan"),
            "abs_px_per_mm_rel_err_median": med([abs(r["px_per_mm_rel_err"]) for r in sel]),
            "density_abs_mae_median": med([r["density_abs_mae"] for r in sel]),
            "density_diff_mae_median": med([r["density_diff_mae"] for r in sel]),
            "margin_db_median": med([r["margin_db"] for r in sel]),
            "three_point_frac": float(np.mean([r["tone_method"] == "three_point" for r in sel])),
        }
        for key in [k for k in sel[0] if k.startswith("floor_")]:
            per_arm[arm][f"{key}_median"] = med([r[key] for r in sel])

    base = per_arm.get("baseline", {})
    oracle = per_arm.get("oracle", {})
    deltas = {}
    for arm in ("framing", "wedge", "ruler", "both"):
        a = per_arm.get(arm)
        if not a or not base:
            continue
        deltas[arm] = {
            "d_abs_gamma_err": a["abs_gamma_err_median"] - base["abs_gamma_err_median"],
            "d_abs_px_per_mm_rel_err": (a["abs_px_per_mm_rel_err_median"]
                                        - base["abs_px_per_mm_rel_err_median"]),
            "d_density_abs_mae": a["density_abs_mae_median"] - base["density_abs_mae_median"],
            "d_density_diff_mae": a["density_diff_mae_median"] - base["density_diff_mae_median"],
            "d_margin_db": a["margin_db_median"] - base["margin_db_median"],
        }
        # How much of the gap to a perfect fix this aid actually closed. Negative
        # means it moved away from the oracle -- worth knowing, and the reason the
        # oracle arm is run at all.
        for metric, key in (("density_abs_mae", "closed_frac_density_abs"),
                            ("density_diff_mae", "closed_frac_density_diff")):
            gap = base[f"{metric}_median"] - oracle.get(f"{metric}_median", float("nan"))
            got = base[f"{metric}_median"] - a[f"{metric}_median"]
            deltas[arm][key] = float(got / gap) if np.isfinite(gap) and abs(gap) > 1e-9 else float("nan")

    # Verdict churn: does an aid change what the certificate says, not just the
    # numbers behind it? A clinic feels this and nothing else.
    by_case: dict = {}
    for r in rows:
        by_case.setdefault((r["image"], r["severity"], r["gamma_true"], r["collimation"]), {})[r["arm"]] = r
    flips = {}
    for arm in ("framing", "wedge", "ruler", "both", "oracle"):
        n = flipped = 0
        for case in by_case.values():
            if "baseline" in case and arm in case:
                n += 1
                flipped += case["baseline"]["verdict"] != case[arm]["verdict"]
        flips[arm] = {"n_cases": n, "verdict_flips": flipped,
                      "flip_rate": float(flipped / n) if n else float("nan")}
    return {"per_arm": per_arm, "delta_vs_baseline": deltas, "verdict_flips": flips}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/fiducial_value.json")
    ap.add_argument("--images", type=int, default=6)
    ap.add_argument("--size", type=int, default=640,
                    help="canonical film resolution; 640 keeps a run to minutes and is enough "
                         "for the wedge's steps to be several pixels across")
    ap.add_argument("--severities", default="0.0,0.35,0.7")
    ap.add_argument("--gammas", default="1.8,2.2,2.6,3.0",
                    help="true ISP exponents to sweep. The prior is 2.2, so this spans "
                         "'the prior happens to be right' to 'the prior is badly wrong'")
    ap.add_argument("--collimations", default="0.03,0.06,0.10",
                    help="collimation margin as a fraction of the film's short side. This is "
                         "what makes the assumed cassette-diagonal scale wrong in the field")
    ap.add_argument("--film-long-mm", type=float, default=432.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="2 images, 1 severity, 2 gammas")
    args = ap.parse_args()

    severities = [float(v) for v in args.severities.split(",") if v.strip()]
    gammas = [float(v) for v in args.gammas.split(",") if v.strip()]
    collimations = [float(v) for v in args.collimations.split(",") if v.strip()]
    n_images = args.images
    if args.quick:
        n_images, severities, gammas, collimations = 2, [0.35], [1.8, 3.0], [0.06]

    film, spec, findings = FilmModel(), FloorSpec(), FIND.core()
    rng = np.random.default_rng(args.seed)

    iso = fit_isolation()
    print("fit_tone in isolation (anchors from its own model, 8-bit quantised only):")
    for r in iso:
        print(f"  gamma {r['gamma_true']:.1f}, ISP black {r['black_level']:.2f} -> "
              f"{r['gamma_fitted']:.3f}  err {r['error']:+.3f}  sigma {r['sigma']:.2f}  "
              f"({r['method']})")
    zero_black = max(abs(r["error"]) for r in iso if r["black_level"] == 0.0)
    with_black = max(abs(r["error"]) for r in iso if r["black_level"] > 0.0)
    print(f"  worst |error| {zero_black:.3f} with no ISP black level -- the fit itself is "
          f"{'sound' if zero_black < 0.05 else 'SUSPECT'}")
    print(f"  worst |error| {with_black:.3f} once the ISP has a 0.02 pedestal -- that gap is the "
          f"price of pinning the black point, paid to keep the veil measurable\n")

    total = n_images * len(severities) * len(gammas) * len(collimations)
    print(f"{total} photographs x {len(ARMS)} arms")
    rows: list[dict] = []
    done = 0
    for i in range(n_images):
        for s in severities:
            for g in gammas:
                for c in collimations:
                    rows.extend(run_case(i, s, g, c, args.size, args.film_long_mm,
                                         film, spec, findings, rng))
                    done += 1
                    if done % 5 == 0 or done == total:
                        print(f"  {done}/{total}", flush=True)

    summary = summarise(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fit_isolation": iso, "summary": summary, "rows": rows,
                               "config": vars(args)}, indent=2))

    pa = summary["per_arm"]
    print("\n" + "=" * 78)
    print(f"{'arm':<10} {'|gamma err|':>11} {'2sig cov':>9} {'|px/mm err|':>12} "
          f"{'D abs MAE':>11} {'D diff MAE':>12} {'margin dB':>10}")
    print("-" * 78)
    for arm in ARMS:
        a = pa.get(arm)
        if not a:
            continue
        print(f"{arm:<10} {a['abs_gamma_err_median']:>11.3f} {a['gamma_2sigma_coverage']:>9.2f} "
              f"{a['abs_px_per_mm_rel_err_median']:>12.3f} {a['density_abs_mae_median']:>11.4f} "
              f"{a['density_diff_mae_median']:>12.4f} {a['margin_db_median']:>10.2f}")
    print("-" * 78)
    for arm, d in summary["delta_vs_baseline"].items():
        print(f"{arm:<10} vs baseline: gamma {d['d_abs_gamma_err']:+.3f}, "
              f"px/mm {d['d_abs_px_per_mm_rel_err']:+.3f}, "
              f"D_abs {d['d_density_abs_mae']:+.4f} "
              f"(closes {100 * d['closed_frac_density_abs']:.0f}% of the oracle gap), "
              f"margin {d['d_margin_db']:+.2f} dB")
    print("-" * 78)
    for arm, f in summary["verdict_flips"].items():
        print(f"{arm:<10} certificate verdict changes on {f['verdict_flips']}/{f['n_cases']} "
              f"photographs ({f['flip_rate']:.2f})")
    print("=" * 78)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
