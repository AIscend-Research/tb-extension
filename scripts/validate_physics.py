#!/usr/bin/env python3
"""Run the experiments that could falsify the physics track. No data, no GPU.

Three checks, in the order they should be believed:

1. **Fiducial recovery** -- does the detector find the marker and the collimation
   corners where they actually are?
2. **Channel recovery** -- handed only an 8-bit JPEG, does the blind estimator
   recover the veil fraction, the PSF width and the density map that generated it?
3. **Detectability** -- and the one that matters: does the density resolution floor
   predict the contrast at which an *optimal* detector starts to fail? This is the
   claim that makes the certificate a physical bound rather than another confidence
   score, and it is free to come out wrong.

    python scripts/validate_physics.py --quick        # ~1 minute
    python scripts/validate_physics.py                # ~10 minutes, publication run

Writes CSVs and a summary JSON to --out. Exit code is 1 if the detectability
calibration falls outside the tolerance band, so this can gate CI.

Interpreting the calibration ratio
----------------------------------
`ratio = predicted_floor / empirical_threshold`. Above 1 the bound is conservative
-- it declares information lost slightly before an optimal detector actually
loses it. Below 1 it is optimistic, which is the dangerous direction, because the
certificate would pass a photograph that cannot carry the finding. Report the
median ratio with the results; a bound quoted without its measured calibration is
just an assertion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from tbtrust.physics import validate as V
from tbtrust.physics.findings import CORE_FINDINGS

# Developed chest film spans roughly OD 0.2-3.0 (base+fog to the darkest lung
# field). A predicted floor above this is not a statement about the photograph;
# it is the inversion having diverged, and it must not be averaged in as though
# it were a measurement.
MAX_PLAUSIBLE_OD = 4.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/physics_validation")
    ap.add_argument("--quick", action="store_true", help="small run for CI / a smoke check")
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--trials", type=int, default=None, help="paired captures per contrast level")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=None,
                    help="number of seeds to aggregate the detectability test over "
                         "(default 5, or 1 with --quick). A single seed cannot "
                         "establish the calibration ratio -- see --help notes.")
    ap.add_argument("--tolerance", type=float, nargs=2, default=(0.4, 3.0),
                    help="acceptable band for the median predicted/empirical ratio")
    ap.add_argument("--max-spread", type=float, default=2.0,
                    help="max allowed ratio between the best and worst per-seed median")
    ap.add_argument("--max-degenerate", type=float, default=0.1,
                    help="max allowed fraction of conditions yielding no usable ratio")
    args = ap.parse_args()

    quick = args.quick
    size = args.size or (192 if quick else 320)
    trials = args.trials or (8 if quick else 24)
    n_seeds = args.seeds if args.seeds is not None else (1 if quick else 5)
    seeds = [args.seed + i for i in range(max(1, n_seeds))]
    n_images = 2 if quick else 8
    severities = (0.0, 0.5) if quick else (0.0, 0.25, 0.5, 0.75, 1.0)
    det_sevs = (0.0, 0.5) if quick else (0.0, 0.25, 0.5, 0.75)
    findings = ("infiltrate",) if quick else CORE_FINDINGS

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"config": {"size": size, "trials": trials, "n_images": n_images,
                                "severities": list(severities), "quick": quick,
                                "seeds": seeds}}

    # ---------------------------------------------------------------- 1. fiducials
    print("1/3  fiducial recovery ...", flush=True)
    fid = pd.DataFrame(V.fiducial_recovery(n_images=n_images, size=size, seed=args.seed,
                                           severities=severities))
    fid.to_csv(out / "fiducial_recovery.csv", index=False)
    summary["fiducials"] = {
        "coverage_full_frac": float((fid["coverage"] == "full").mean()),
        "marker_iou_median": float(fid["marker_iou"].median(skipna=True)),
        "corner_err_px_median": float(fid["corner_err_px"].median(skipna=True)),
        "usable_mtf_edge_frac": float((fid["n_mtf_edges"] > 0).mean()),
    }
    print("     " + json.dumps(summary["fiducials"]))

    # ----------------------------------------------------------------- 2. channel
    print("2/3  channel recovery ...", flush=True)
    rec = pd.DataFrame(V.recovery_experiment(n_images=n_images, severities=severities,
                                             size=size, seed=args.seed))
    rec.to_csv(out / "channel_recovery.csv", index=False)
    summary["channel"] = V.summarize_recovery(rec.to_dict("records"))
    print("     " + json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                for k, v in summary["channel"].items()}))

    # ----------------------------------------------------------- 3. detectability
    print("3/3  detectability (the falsification test) ...", flush=True)
    det_rows = []
    for sd in seeds:
        for sev in det_sevs:
            for f in findings:
                r = V.detectability_experiment(severity=sev, finding=f, n_trials=trials,
                                               size=size, seed=sd)
                row = r.as_dict()
                row["seed"] = sd
                # Two ways a condition can fail to yield a usable ratio, kept apart
                # because they mean different things and the old code silently
                # merged both into "not > 0" and dropped them from the median:
                #
                # degenerate  -- the matched filter got zero discrimination at every
                #                contrast (slope <= 0), so empirical_threshold is inf
                #                and ratio is 0. That is the detector reporting total
                #                failure, which is the strongest possible evidence
                #                against the bound, so excluding it inverts its meaning.
                # unphysical  -- the inversion returned a floor no developed film can
                #                exhibit. Chest radiographs span roughly OD 0.2-3.0;
                #                a floor above MAX_PLAUSIBLE_OD means the estimator
                #                diverged, not that the photograph is bad.
                row["degenerate"] = bool(r.dprime_slope <= 0 or not np.isfinite(r.empirical_threshold))
                row["unphysical"] = bool(r.predicted_floor > MAX_PLAUSIBLE_OD)
                det_rows.append(row)
                flag = "DEGEN" if row["degenerate"] else ("UNPHYS" if row["unphysical"] else
                                                          ("PASS" if r.passes else "fail"))
                print(f"     seed={sd} sev={sev:<5} {f:<16} predicted={r.predicted_floor:8.4f} "
                      f"empirical={r.empirical_threshold:8.4f} ratio={r.ratio:6.2f} "
                      f"r2={r.linearity_r2:5.2f} {flag}", flush=True)
    det = pd.DataFrame(det_rows)
    det.to_csv(out / "detectability.csv", index=False)

    usable = det[np.isfinite(det["ratio"]) & (det["ratio"] > 0)
                 & ~det["degenerate"] & ~det["unphysical"]]
    med = float(usable["ratio"].median()) if len(usable) else float("nan")

    # Per-seed medians. A single run's median is not a measurement of anything:
    # the spread across seeds is what says whether the number can be quoted.
    per_seed = {int(sd): (float(g["ratio"].median()) if len(g) else float("nan"))
                for sd, g in usable.groupby("seed")}
    spread = (max(per_seed.values()) / min(per_seed.values())
              if per_seed and min(per_seed.values()) > 0 else float("inf"))

    summary["detectability"] = {
        "n_conditions": len(det),
        "n_usable": len(usable),
        "n_degenerate": int(det["degenerate"].sum()),
        "n_unphysical": int(det["unphysical"].sum()),
        "degenerate_frac": float(det["degenerate"].mean()),
        "median_ratio": med,
        "iqr_ratio": [float(usable["ratio"].quantile(0.25)), float(usable["ratio"].quantile(0.75))]
        if len(usable) > 1 else None,
        "median_ratio_per_seed": per_seed,
        "across_seed_spread": float(spread),
        "pass_frac": float(det["passes"].mean()),
        "median_linearity_r2": float(usable["linearity_r2"].median()) if len(usable) else float("nan"),
    }

    # --------------------------------------------------------------- 4. ordering
    print("     certificate ordering ...", flush=True)
    cons = pd.DataFrame(V.certificate_consistency(severities=severities,
                                                  n_images=max(2, n_images // 2), size=size,
                                                  seed=args.seed))
    cons.to_csv(out / "certificate_consistency.csv", index=False)
    by_sev = cons.groupby("severity")["margin_db"].median()
    monotone = bool(np.all(np.diff(by_sev.to_numpy()) <= 1e-9))
    summary["ordering"] = {
        "margin_db_by_severity": {float(k): float(v) for k, v in by_sev.items()},
        "margin_falls_monotonically": monotone,
        "insufficient_frac_by_severity": {
            float(k): float((g["certificate"] == "insufficient").mean())
            for k, g in cons.groupby("severity")
        },
    }

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lo, hi = args.tolerance
    d = summary["detectability"]
    in_band = bool(np.isfinite(med) and lo <= med <= hi)
    stable = bool(np.isfinite(spread) and spread <= args.max_spread)
    complete = bool(d["degenerate_frac"] <= args.max_degenerate and d["n_unphysical"] == 0)
    ok = in_band and stable and complete and monotone

    print("\n" + "=" * 66)
    print(f"median predicted/empirical ratio : {med:.2f}   (tolerance {lo}-{hi})  {'ok' if in_band else 'FAIL'}")
    print("per-seed medians                 : "
          + ", ".join(f"{s}:{v:.2f}" for s, v in sorted(per_seed.items())))
    print(f"across-seed spread (max/min)     : {spread:.2f}   (max {args.max_spread})  "
          f"{'ok' if stable else 'FAIL'}")
    print(f"conditions with no usable ratio  : {d['n_degenerate']}/{d['n_conditions']} degenerate, "
          f"{d['n_unphysical']} unphysical  {'ok' if complete else 'FAIL'}")
    print(f"certificate margin monotone      : {monotone}")
    print(f"conditions passing individually  : {d['pass_frac']:.0%}")

    if not stable:
        print("\nWARNING: the calibration ratio is NOT REPRODUCIBLE across seeds. No single\n"
              "run's median can be quoted as the bound's calibration until this is fixed --\n"
              "the number would be an artefact of the seed, not a measurement.")
    if not complete:
        print("\nWARNING: some conditions produced no usable ratio at all (zero detector\n"
              "discrimination, or a floor above what any film can exhibit). These are the\n"
              "estimator's worst failures, so excluding them from the median would report\n"
              "a bound that looks better the harder it fails.")
    if in_band and stable and complete and med > 1:
        print("\nthe bound is conservative -- it calls information lost slightly before an\n"
              "optimal detector actually loses it, which is the safe direction")
    elif in_band and stable and complete:
        print("\nWARNING: the bound is OPTIMISTIC -- it certifies photographs an optimal\n"
              "detector cannot actually read. Do not deploy the certificate as a safety\n"
              "valve until this is above 1.")
    print(f"\nwrote {out}/")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
