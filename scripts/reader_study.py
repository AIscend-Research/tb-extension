#!/usr/bin/env python3
"""Draw the reader-study sample, and measure what the study can and cannot see.

`docs/LIMITATIONS.md` §4 says the second-reader framing closes when "a
radiologist rates a sample of held-out images for 'would you seek a second
opinion', and that is correlated against predicted uncertainty". This script is
everything in that sentence except the radiologist:

    design    draw a balanced sample over (physics margin x learned uncertainty),
              emit the blinded rating sheet, the unblinding key, and the design's
              own numbers -- how discordant the two signals actually are on this
              corpus, which is what the study has to have to be worth running.
    power     the reader-noise ceiling and the empirical power of the
              pre-registered test, over a grid of n, reader count and effect
              size. Run before booking anyone.
    analyze   the pre-registered read-out, once ratings come back.

The design numbers matter more than they look. A reader study on a corpus where
the certificate and the classifier agree everywhere cannot distinguish them no
matter how many films it buys: every discordant cell would be empty. So the
first thing reported is the joint distribution, and the first thing to check is
that the off-diagonal cells are populated at all.

    python scripts/reader_study.py design  --rows outputs/physics_deferral_rows.csv
    python scripts/reader_study.py power   --out outputs/reader_study_power.csv
    python scripts/reader_study.py analyze --key outputs/reader_study_key.csv \
                                           --ratings ratings_reader1.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from tbtrust.eval import reader_study as RS

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------

def _load_rows(path: Path, test_only: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if test_only and "is_test" in df.columns:
        df = df[df["is_test"].astype(bool)]
    need = ["margin_db", "mc_std", "key", "severity"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing {missing}; run scripts/physics_deferral_real.py")
    return df.reset_index(drop=True)


def _export_images(design, outdir: Path, size: int, seed: int) -> dict:
    """Render the exact photographs the rows were scored on, named by case_id.

    Regenerated rather than stored: `utils.seed.capture_seed` is CRC32 over
    (path, severity, seed), so the same call in a later process reproduces the
    same capture bit for bit. That is the whole reason the study can be blinded
    -- the reader sees a file called C337134.png and nothing about it says which
    clinic it came from, what severity was applied, or what the model thought.
    """
    from PIL import Image

    from tbtrust.physics.film import simulate
    from tbtrust.utils.seed import capture_seed

    outdir.mkdir(parents=True, exist_ok=True)
    written, failed = 0, []
    for r in design.rows:
        path = r.get("path")
        if not path or not Path(path).exists():
            failed.append(r["case_id"])
            continue
        img = Image.open(path).convert("L")
        if max(img.size) != size:
            img = img.resize((size, size), Image.BILINEAR)
        sev = float(r.get("severity", 0.0))
        photo, _ = simulate(np.asarray(img), severity=sev,
                            rng=np.random.default_rng(capture_seed(path, sev, seed)),
                            size=size)
        Image.fromarray(np.asarray(photo, dtype=np.uint8)).save(
            outdir / f"{r['case_id']}.png")
        written += 1
    return {"written": written, "missing_source": failed}


def cmd_design(args) -> int:
    df = _load_rows(Path(args.rows), args.test_only)
    df["physics_score"] = RS.physics_referral_score(
        df["margin_db"], df["abstained"] if "abstained" in df.columns else None)
    rows = df.to_dict("records")
    design = RS.stratified_sample(
        rows, n_cases=args.n_cases, uncertainty_col=args.uncertainty,
        repeat_fraction=args.repeat_fraction, seed=args.seed)

    sheet, key = RS.rating_sheet(design)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sheet).to_csv(outdir / "reader_study_sheet.csv", index=False)
    pd.DataFrame(key).to_csv(outdir / "reader_study_key.csv", index=False)

    # --- what the corpus itself says about whether the study can separate the
    # --- two signals. Discordance is the study's whole budget, and two of the
    # --- binary decisions turn out to be constant on this corpus (see below),
    # --- which is a design finding rather than a bug: a contrast against a
    # --- constant is unmeasurable no matter how many readers are booked.
    m = df["margin_db"].to_numpy(float)
    u = df[args.uncertainty].to_numpy(float)
    ps = df["physics_score"].to_numpy(float)
    abstain = ~np.isfinite(m)
    if "abstained" in df.columns:
        abstain |= df["abstained"].astype(bool).to_numpy()

    verdicts = {}
    for c in [c for c in df.columns if c.startswith("verdict_")]:
        v = df[c].astype("string")
        verdicts[c[len("verdict_"):]] = {
            "insufficient": float((v == "insufficient").mean()),
            "marginal": float((v == "marginal").mean()),
            "detectable": float((v == "detectable").mean()),
            "abstain": float(v.isna().mean()),
        }

    degenerate = {k: v for k, v in {
        "triage_action": df["triage_action"].nunique() if "triage_action" in df else None,
        "model_confident": df["model_confident"].nunique() if "model_confident" in df else None,
        "worst_finding_verdict": int(df.get("certificate", pd.Series(dtype=str))
                                      .nunique()) if "certificate" in df else None,
    }.items() if v is not None and v <= 1}

    # Matched-rate discordance on the *continuous* signals: flag the top q of
    # each and ask how often the two flag different photographs. q is set from
    # the one per-finding verdict that is not degenerate, so the comparison is
    # made at an operating point the certificate actually uses.
    q = verdicts.get("infiltrate", {}).get("insufficient", 0.33) or 0.33
    p_hi = ps >= np.quantile(ps, 1 - q)
    u_hi = u >= np.quantile(u, 1 - q)
    joint = {
        "n_rows": len(df),
        "n_films": int(df["key"].nunique()),
        "physics_abstain_rate": float(abstain.mean()),
        "worst_finding_insufficient_rate": float(
            abstain.mean() + np.nan_to_num(m <= 0.0, nan=0.0).mean()),
        "per_finding_verdicts": verdicts,
        "degenerate_columns": degenerate,
        "matched_flag_rate": float(q),
        "both_flag": float((p_hi & u_hi).mean()),
        "physics_only": float((p_hi & ~u_hi).mean()),
        "learned_only": float((~p_hi & u_hi).mean()),
        "neither": float((~p_hi & ~u_hi).mean()),
        "spearman_margin_vs_uncertainty": float(
            pd.Series(m).rank().corr(pd.Series(u).rank())),
        "spearman_physics_score_vs_uncertainty": float(
            pd.Series(ps).rank().corr(pd.Series(u).rank())),
    }

    cells = pd.DataFrame([
        {"cell": c, "population": design.population[c],
         "drawn": design.allocation.get(c, 0),
         "sampling_weight": design.population[c] / max(1, design.allocation.get(c, 0))}
        for c in sorted(design.population)
    ])
    cells.to_csv(outdir / "reader_study_strata.csv", index=False)

    images = None
    if args.export_images:
        images = _export_images(design, outdir / "reader_study_images",
                               size=args.size, seed=args.capture_seed)
        print(f"  wrote {images['written']} blinded PNGs to "
              f"{outdir}/reader_study_images/")

    summary = {
        "rows_source": str(args.rows),
        "images": images,
        "test_only": bool(args.test_only),
        "uncertainty_col": args.uncertainty,
        "n_cases_drawn": design.n_cases,
        "n_shown_including_repeats": design.n_shown,
        "n_repeats": design.n_repeats,
        "strata": design.n_strata,
        "seed": design.seed,
        "signal_overlap": joint,
        "cells": cells.to_dict("records"),
    }
    (outdir / "reader_study_design.json").write_text(json.dumps(summary, indent=2))

    print(f"drew {design.n_cases} films, {design.n_shown} shown "
          f"({design.n_repeats} repeats) over {len(design.allocation)} cells")
    print(f"  certificate abstains      {joint['physics_abstain_rate']:.3f}")
    print(f"  matched flag rate q       {joint['matched_flag_rate']:.3f}")
    print(f"  discordant (physics only) {joint['physics_only']:.3f}")
    print(f"  discordant (learned only) {joint['learned_only']:.3f}")
    print(f"  rank corr physics vs unc  "
          f"{joint['spearman_physics_score_vs_uncertainty']:+.3f}")
    if degenerate:
        print(f"  WARNING constant columns  {sorted(degenerate)} -- any binary "
              f"contrast against these is unmeasurable on this corpus")
    print(f"  -> {outdir}/reader_study_sheet.csv (blinded), _key.csv, _design.json")
    return 0


def cmd_power(args) -> int:
    ns = [int(x) for x in args.n_cases.split(",")]
    iccs = [float(x) for x in args.icc.split(",")]
    readers = [int(x) for x in args.readers.split(",")]
    effects = [float(x) for x in args.effect.split(",")]

    rows = []
    for icc in iccs:
        for k in readers:
            ceil = RS.reader_noise_ceiling(
                n_cases=max(ns), model=RS.ReaderModel(icc_single=icc, n_readers=k,
                                                      refer_rate=args.refer_rate),
                n_sim=args.n_sim, seed=args.seed)
            for n in ns:
                for r in effects:
                    p = RS.design_power(
                        n_cases=n,
                        model=RS.ReaderModel(icc_single=icc, n_readers=k, signal_r=r,
                                             refer_rate=args.refer_rate),
                        n_sim=args.n_sim, n_boot=args.n_boot, seed=args.seed)
                    rows.append({
                        "n_cases": n, "n_readers": k, "icc_single": icc,
                        "signal_r": r, "power": p["power"],
                        "type_i_at_null": p["type_i_at_null"],
                        "auc_ceiling_single_reader": ceil["auc_ceiling_single_reader"],
                        "auc_ceiling_majority_vote": ceil["auc_ceiling_majority_vote"],
                        "consensus_reliability": ceil["consensus_reliability"],
                        "min_detectable_auc": RS.minimum_detectable_auc(
                            n, refer_rate=args.refer_rate),
                    })
                    print(f"n={n:4d} k={k} icc={icc:.2f} r={r:.2f}  "
                          f"power={p['power']:.2f}  typeI={p['type_i_at_null']:.3f}  "
                          f"ceiling={ceil['auc_ceiling_majority_vote']:.3f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")
    return 0


def cmd_analyze(args) -> int:
    key = pd.read_csv(args.key)
    sheets = [pd.read_csv(p) for p in args.ratings]
    for i, s in enumerate(sheets):
        if "case_id" not in s.columns or "refer_1_to_5" not in s.columns:
            raise SystemExit(f"{args.ratings[i]}: expected case_id + refer_1_to_5")

    merged = key.set_index("case_id")
    reals = merged[~merged.index.str.endswith("R")]
    refer, ordinal = [], []
    for s in sheets:
        s = s.set_index("case_id").reindex(reals.index)
        o = pd.to_numeric(s["refer_1_to_5"], errors="coerce")
        ordinal.append(o.to_numpy(float))
        refer.append((o >= RS.REFER_BINARY_CUT).to_numpy())
    refer = np.column_stack(refer)
    ordinal = np.column_stack(ordinal)
    if np.isnan(ordinal).any():
        raise SystemExit("missing ratings: every drawn case must be rated by every reader")

    reals = reals.assign(physics_score=RS.physics_referral_score(
        reals["margin_db"],
        reals["abstained"] if "abstained" in reals.columns else None))
    rows = reals.reset_index().to_dict("records")
    defers = None
    if "model_confident" in reals.columns:
        defers = ~reals["model_confident"].astype(bool).to_numpy()
    adequacy = physics_retake = None
    if all("adequacy" in s.columns for s in sheets) and "triage_action" in reals.columns:
        ad = np.column_stack([
            (pd.read_csv(p).set_index("case_id").reindex(reals.index)["adequacy"]
             .astype(str).str.strip().str.startswith("inadequate")).to_numpy()
            for p in args.ratings])
        adequacy = ad.mean(axis=1) > 0.5
        physics_retake = (reals["triage_action"].astype(str) == "retake").to_numpy()

    res = RS.analyze(rows, refer, ordinal=ordinal, model_defers=defers,
                     adequacy_inadequate=adequacy, physics_says_retake=physics_retake,
                     n_boot=args.n_boot, seed=args.seed)
    ceil = RS.reader_noise_ceiling(
        n_cases=len(rows),
        model=RS.ReaderModel(
            icc_single=res["inter_reader"].get("icc_single_reader", 0.5),
            n_readers=refer.shape[1], refer_rate=res["refer_rate_raw"]),
        n_sim=args.n_sim)
    res["measured_ceiling"] = ceil

    Path(args.out).write_text(json.dumps(res, indent=2, default=float))
    for name, r in res["signals"].items():
        flag = "  *" if r["beats_chance"] else ""
        print(f"{name:20s} AUC {r['auc']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]{flag}")
    print(f"{'ceiling (vote)':20s} AUC {ceil['auc_ceiling_majority_vote']:.3f}")
    print(f"-> {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("design", help="draw the sample and emit the sheets")
    d.add_argument("--rows", default="outputs/physics_deferral_rows.csv")
    d.add_argument("--outdir", default="outputs")
    d.add_argument("--n-cases", type=int, default=120)
    d.add_argument("--uncertainty", default="mc_std")
    d.add_argument("--repeat-fraction", type=float, default=0.1)
    d.add_argument("--test-only", action="store_true", default=True)
    d.add_argument("--all-rows", dest="test_only", action="store_false")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--export-images", action="store_true",
                   help="render the blinded photographs the readers actually see")
    d.add_argument("--size", type=int, default=1024,
                   help="capture size; must match scripts/physics_deferral_real.py")
    d.add_argument("--capture-seed", type=int, default=0,
                   help="capture seed used when the rows were produced")
    d.set_defaults(func=cmd_design)

    p = sub.add_parser("power", help="ceiling + empirical power over a grid")
    p.add_argument("--out", default="outputs/reader_study_power.csv")
    p.add_argument("--n-cases", default="60,120,240")
    p.add_argument("--icc", default="0.4,0.5,0.6")
    p.add_argument("--readers", default="1,3")
    p.add_argument("--effect", default="0.30,0.45,0.60")
    p.add_argument("--refer-rate", type=float, default=0.25)
    p.add_argument("--n-sim", type=int, default=200)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_power)

    a = sub.add_parser("analyze", help="the pre-registered read-out")
    a.add_argument("--key", default="outputs/reader_study_key.csv")
    a.add_argument("--ratings", nargs="+", required=True)
    a.add_argument("--out", default="outputs/reader_study_results.json")
    a.add_argument("--n-boot", type=int, default=2000)
    a.add_argument("--n-sim", type=int, default=400)
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(func=cmd_analyze)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
