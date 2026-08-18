#!/usr/bin/env python3
"""How much of the veil error is interior extrapolation? Measure it, don't assume.

The beam stop is an annulus, so `glare.estimate_veil` fits a polynomial surface to
probes that all sit around the *rim* of the collimated field, then evaluates that
surface across the interior where there are no probes at all. Every write-up of
this project has called that the leading known optimism in the bound. Nobody has
measured how large it is.

This does. On synthetic captures the true veil field is known by construction
(`CaptureTruth.glare_field_true`), so the veil error can be split into:

* **annulus**  -- inside `beamstop_mask`, where the probes actually are. This is
  interpolation, and it is the estimator's best case.
* **interior** -- inside the collimated field but away from the beam stop. This is
  pure extrapolation, and it is where a specular reflection of a window would sit.

If the two are comparable, the annulus geometry is fine and the veil error lives
somewhere else. If interior error is much larger, and grows with severity, then
the extrapolation is the defect and a second probe is worth building.

    python scripts/measure_interior_glare.py --out outputs/interior_glare

No data, no GPU. Writes a per-(image, severity) CSV and a summary JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from tbtrust.physics import _ops
from tbtrust.physics.film import FilmModel, capture, sample_params, synthetic_chest_density
from tbtrust.physics.invert import invert

EPS = 1e-9


def _frac_field(veil: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Veil as a fraction of signal, the quantity the floor actually consumes."""
    return np.asarray(veil, dtype=np.float64) / np.maximum(np.asarray(signal, dtype=np.float64), EPS)


def _normalised(veil: np.ndarray, ref: float) -> np.ndarray:
    """Veil in units of a single reference luminance, shared by both regions.

    Normalising by the local `signal` is wrong for this comparison: inside the
    beam stop the signal is ~zero by construction -- that is what a beam stop is --
    so veil/signal there runs to tens or hundreds and swamps any real difference.
    Dividing both regions by one scalar (the median true signal over the field
    interior) keeps the numbers dimensionless and comparable, which is the whole
    point of putting annulus and interior side by side.
    """
    return np.asarray(veil, dtype=np.float64) / max(float(ref), EPS)


def _stats(true_f: np.ndarray, est_f: np.ndarray, mask: np.ndarray) -> dict:
    if mask is None or not mask.any():
        return {"n_px": 0, "true": np.nan, "est": np.nan, "err": np.nan, "rel_err": np.nan}
    t = float(np.median(true_f[mask]))
    e = float(np.median(est_f[mask]))
    return {
        "n_px": int(mask.sum()),
        "true": t,
        "est": e,
        "err": e - t,
        # Negative means the estimator under-reports the veil, which is the
        # direction that makes the floor too low and the certificate optimistic.
        "rel_err": (e - t) / max(abs(t), EPS),
    }


def run(n_images: int, severities: tuple[float, ...], size: int, seed: int,
        guard_px: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    film = FilmModel()
    rows: list[dict] = []

    for i in range(n_images):
        base, ftruth = synthetic_chest_density(size=size, rng=np.random.default_rng(1000 + i),
                                               film=film)
        for s in severities:
            params = sample_params(float(s), rng)
            photo, truth = capture(base, params, fiducial_truth=ftruth, film=film,
                                   rng=np.random.default_rng(rng.integers(1 << 31)))
            if truth.glare_field_true is None or truth.signal_true is None:
                continue
            cal = invert(photo, film=film)

            fid = cal.fiducials
            bs = fid.beamstop_mask
            fm = fid.field_mask
            if bs is None or fm is None or not bs.any():
                rows.append({"image": i, "severity": float(s),
                             "coverage": fid.coverage.value, "usable": False})
                continue

            # Interior = inside the collimated field, clear of the beam stop by a
            # guard band so PSF bleed from the rim probes is not counted as
            # interior. That guard is what makes this extrapolation rather than
            # interpolation-with-a-margin.
            interior = fm & ~_ops.binary_dilate(bs, guard_px)
            annulus = bs & fm if (bs & fm).any() else bs
            if not interior.any():
                rows.append({"image": i, "severity": float(s),
                             "coverage": fid.coverage.value, "usable": False})
                continue

            # One reference luminance for both regions -- see _normalised.
            ref = float(np.median(np.asarray(truth.signal_true, dtype=np.float64)[interior]))
            true_f = _normalised(truth.glare_field_true, ref)
            est_f = _normalised(cal.veil, ref)

            a = _stats(true_f, est_f, annulus)
            it = _stats(true_f, est_f, interior)
            rows.append({
                "image": i,
                "severity": float(s),
                "coverage": fid.coverage.value,
                "usable": True,
                "glare_method": cal.glare.method,
                "specular_frac_true": float(params.specular_strength)
                if hasattr(params, "specular_strength") else np.nan,
                "annulus_n_px": a["n_px"], "annulus_true": a["true"],
                "annulus_est": a["est"], "annulus_err": a["err"],
                "annulus_rel_err": a["rel_err"],
                "interior_n_px": it["n_px"], "interior_true": it["true"],
                "interior_est": it["est"], "interior_err": it["err"],
                "interior_rel_err": it["rel_err"],
                # The headline: how much worse is extrapolation than interpolation?
                "extrapolation_penalty": (abs(it["rel_err"]) - abs(a["rel_err"]))
                if np.isfinite(it["rel_err"]) and np.isfinite(a["rel_err"]) else np.nan,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/interior_glare")
    ap.add_argument("--n-images", type=int, default=12)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guard-px", type=int, default=12,
                    help="dilate the beam stop by this much before calling the rest interior")
    ap.add_argument("--severities", default="0.0,0.25,0.5,0.75,1.0")
    args = ap.parse_args()

    sevs = tuple(float(x) for x in args.severities.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = run(args.n_images, sevs, args.size, args.seed, args.guard_px)
    df = pd.DataFrame(rows)
    df.to_csv(out / "interior_glare.csv", index=False)

    ok = df[df["usable"]] if "usable" in df else df
    if ok.empty:
        print("no usable captures (no beam stop detected)", file=sys.stderr)
        return 1

    print(f"\n{len(ok)} usable captures of {len(df)}\n")
    print(f"  {'severity':>9} {'annulus err':>12} {'interior err':>13} {'penalty':>9}")
    print("  " + "-" * 47)
    summary = {}
    for s, g in ok.groupby("severity"):
        a = float(g["annulus_rel_err"].median())
        i = float(g["interior_rel_err"].median())
        p = float(g["extrapolation_penalty"].median())
        summary[float(s)] = {"annulus_rel_err": a, "interior_rel_err": i,
                             "extrapolation_penalty": p, "n": int(len(g))}
        print(f"  {s:>9.2f} {a:>12.3f} {i:>13.3f} {p:>9.3f}")

    a_all = float(ok["annulus_rel_err"].median())
    i_all = float(ok["interior_rel_err"].median())
    (out / "summary.json").write_text(json.dumps(
        {"by_severity": summary, "annulus_rel_err_median": a_all,
         "interior_rel_err_median": i_all,
         "guard_px": args.guard_px, "n_images": args.n_images, "size": args.size},
        indent=2))

    print(f"\n  overall: annulus {a_all:+.3f}   interior {i_all:+.3f}")
    if np.isfinite(a_all) and np.isfinite(i_all) and abs(i_all) > 1.5 * abs(a_all):
        print("\n  The interior is materially worse than the annulus: the veil error is\n"
              "  dominated by extrapolating a rim-only fit across the field. A second\n"
              "  interior probe would attack the right term.")
    else:
        print("\n  Interior and annulus errors are comparable: the annulus geometry is not\n"
              "  the dominant defect, and a second probe would not fix the veil error.")
    print(f"\nwrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
