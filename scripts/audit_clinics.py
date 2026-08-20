#!/usr/bin/env python3
"""What a candidate clinic has to survive before the rotation may report it.

The rotation reports two honest holdout folds because only Montgomery and
Shenzhen contain both classes (`docs/LIMITATIONS.md` §3). Adding a third and a
fourth strengthens the cross-site claim more than any modelling change would --
but only if the new source is genuinely a new site, and `data/splits.py` cannot
tell: it keys on the `clinic` column and trusts it.

Three subcommands, in the order they should be run:

    sources    print the survey in `data/sources.py` -- what exists, what it
               costs to get, and which entries are still unverified.
    overlap    perceptual-hash every pair of clinics in the manifest against each
               other. Catches a "new" source that is a re-bundling of one already
               present, which is the failure mode the Kaggle TB aggregate walks
               you into: same images, new filenames, new folder, clean-looking
               leave-one-clinic-out fold.
    confound   for every real and hypothetical two-class fold, ask whether the
               label can be predicted from capture statistics alone. Runs on the
               existing clinics as a baseline, and on the *hybrid* cohorts
               LIMITATIONS §3 proposes -- mixing one source's normals with
               another's positives -- which is the manufactured fold most likely
               to be reported as real.

    python scripts/audit_clinics.py sources
    python scripts/audit_clinics.py overlap  --manifest data/processed/manifest.csv
    python scripts/audit_clinics.py confound --manifest data/processed/manifest.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from tbtrust.data import audit as A
from tbtrust.data import sources as S


def cmd_sources(args) -> int:
    rows = S.summary_rows()
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_colwidth", 34):
        print(df[["key", "country", "n_images", "both_classes", "resolution",
                  "access", "verdict"]].to_string(index=False))
    print("\ncandidate additional folds, best value first:")
    for s in S.candidate_folds():
        print(f"\n  {s.name} ({s.country}, n={s.n_images}, {s.verdict})")
        print(f"    {s.why}")
        unver = sorted(k for k, v in s.evidence.items() if v != "verified")
        if unver:
            print(f"    unverified, re-check on download: {', '.join(unver)}")
    print("\nexcluded:")
    for s in S.CANDIDATES:
        if not s.is_candidate_fold:
            print(f"  {s.name}: {s.verdict} -- {s.why.splitlines()[0]}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\n-> {args.out}")
    return 0


def cmd_calibrate(args) -> int:
    """Measure what threshold the duplicate check can actually stand on."""
    df = pd.read_csv(args.manifest)
    clinics = sorted(df["clinic"].unique())
    if len(clinics) < 2:
        raise SystemExit("calibration needs two clinics known to be disjoint")
    rng = np.random.default_rng(args.seed)

    def _sample(c, n):
        p = df[df["clinic"] == c]["path"].tolist()
        k = min(n, len(p))
        return [p[i] for i in rng.choice(len(p), k, replace=False)]

    a, b = clinics[0], clinics[1]
    rows = []
    for size in [int(x) for x in args.sizes.split(",")]:
        res = A.calibrate_threshold(_sample(a, args.n_positive), _sample(a, args.n_negative),
                                    _sample(b, args.n_negative), size=size)
        rows.append(res)
        if not res.get("usable"):
            print(f"  size={size:2d} UNUSABLE: {res['note']}")
            continue
        print(f"  size={size:2d} ({res['n_bits']:3d} bits)  duplicate<= {res['positive_max']}  "
              f"different>= {res['negative_min']}  gap={res['gap_bits']:3d}  "
              f"-> threshold {res['recommended_threshold']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"known_disjoint_pair": [a, b], "results": rows}, indent=2))
    print("\nknown-duplicate control: each image against its own 512 px JPEG re-bundle")
    print(f"known-different control : all pairs between {a} and {b}")
    print(f"-> {out}")
    return 0


def cmd_overlap(args) -> int:
    df = pd.read_csv(args.manifest)
    clinics = sorted(df["clinic"].unique())
    rng = np.random.default_rng(args.seed)

    paths: dict = {}
    for c in clinics:
        p = df[df["clinic"] == c]["path"].tolist()
        if args.limit and len(p) > args.limit:
            p = [p[i] for i in rng.choice(len(p), args.limit, replace=False)]
        paths[c] = p

    rows = []
    for a, b in itertools.combinations(clinics, 2):
        rep = A.find_overlap(paths[a], paths[b], threshold=args.threshold, size=args.size)
        rows.append({"clinic_a": a, "clinic_b": b, "n_a": rep.n_a, "n_b": rep.n_b,
                     "n_pairs": len(rep.pairs),
                     "n_a_overlapping": rep.n_overlapping,
                     "fraction_of_a": rep.fraction_of_a,
                     "verdict": rep.verdict()})
        print(f"  {a:12s} vs {b:12s}  pairs={len(rep.pairs):4d}  "
              f"{rep.fraction_of_a:6.3f} of {a}  -> {rep.verdict()}")
        for pa, pb, d in rep.pairs[:args.show]:
            print(f"      d={d}  {Path(pa).name}  ~  {Path(pb).name}")

    # A source is also allowed to duplicate *itself*: the same study exported
    # twice under two names inflates a fold's n and puts one patient on both
    # sides of a train/val split.
    for c in clinics:
        rep = A.find_overlap(paths[c], paths[c], threshold=args.threshold, size=args.size)
        self_pairs = [p for p in rep.pairs if p[0] != p[1]]
        rows.append({"clinic_a": c, "clinic_b": c + " (self)", "n_a": rep.n_a,
                     "n_b": rep.n_b, "n_pairs": len(self_pairs),
                     "n_a_overlapping": len({p[0] for p in self_pairs}),
                     "fraction_of_a": len({p[0] for p in self_pairs}) / max(rep.n_a, 1),
                     "verdict": "clean" if not self_pairs else "suspect"})
        if self_pairs:
            print(f"  {c}: {len(self_pairs)} within-clinic near-duplicate pairs")
            for pa, pb, d in self_pairs[:args.show]:
                print(f"      d={d}  {Path(pa).name}  ~  {Path(pb).name}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")
    return 0


def _features(df: pd.DataFrame, limit: int, seed: int, resize=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if limit and len(df) > limit:
        df = df.iloc[rng.choice(len(df), limit, replace=False)]
    feats = [A.capture_features(p, resize=resize) for p in df["path"]]
    out = pd.DataFrame(feats)
    out["label"] = df["label"].to_numpy()
    out["clinic"] = df["clinic"].to_numpy()
    out["path"] = df["path"].to_numpy()
    return out


def _confound_rows(df, args, variant: str, resize) -> list[dict]:
    feats = _features(df, args.limit, args.seed, resize=resize)
    rows: list[dict] = []
    # Real clinics: the baseline. Whatever number an honest two-class fold
    # produces here is the yardstick every manufactured fold is read against.
    for c in sorted(feats["clinic"].unique()):
        sub = feats[feats["clinic"] == c]
        if sub["label"].nunique() < 2:
            print(f"  {c:12s} single-class, cannot be a fold")
            rows.append({"fold": c, "variant": variant, "kind": "real", "auc": float("nan"),
                         "verdict": "single_class", "n": len(sub)})
            continue
        res = A.source_confound(sub.drop(columns=["label", "clinic", "path"])
                                .to_dict("records"), sub["label"], seed=args.seed)
        rows.append({"fold": c, "variant": variant, "kind": "real", **res,
                     "verdict": A.confound_verdict(res["auc"])})
        print(f"  {c:12s} real     auc={res['auc']:.3f}  "
              f"top={res.get('top_feature')}  -> {rows[-1]['verdict']}")

    # Hybrid cohorts: one source's normals, another's positives. This is the
    # construction LIMITATIONS §3 offers as the way to manufacture more folds,
    # and the number below is what it costs.
    clinics = sorted(feats["clinic"].unique())
    for neg_c, pos_c in itertools.permutations(clinics, 2):
        neg = feats[(feats["clinic"] == neg_c) & (feats["label"] == 0)]
        pos = feats[(feats["clinic"] == pos_c) & (feats["label"] == 1)]
        if len(neg) < 10 or len(pos) < 10:
            continue
        sub = pd.concat([neg, pos])
        res = A.source_confound(sub.drop(columns=["label", "clinic", "path"])
                                .to_dict("records"), sub["label"], seed=args.seed)
        name = f"{neg_c}-normals + {pos_c}-TB"
        rows.append({"fold": name, "variant": variant, "kind": "hybrid", **res,
                     "verdict": A.confound_verdict(res["auc"])})
        print(f"  {name:34s} hybrid   auc={res['auc']:.3f}  "
              f"top={res.get('top_feature')}  -> {rows[-1]['verdict']}")

    return rows


def cmd_confound(args) -> int:
    df = pd.read_csv(args.manifest)
    rows = []
    # Two variants, and the second is the one that matters. On the originals a
    # confound can ride entirely on image dimensions, which the training pipeline
    # resamples away before the network sees anything. Only what survives the
    # resample is a statement about the model.
    variants = [("original", None)]
    if args.resize:
        variants.append((f"resized_{args.resize}", args.resize))
    for variant, resize in variants:
        print(f"\n[{variant}]")
        rows += _confound_rows(df, args, variant, resize)


    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    flat = [{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()}
            for r in rows]
    pd.DataFrame(flat).to_csv(out, index=False)
    print(f"-> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sources", help="the survey of candidate sources")
    s.add_argument("--out", default="outputs/clinic_sources.json")
    s.set_defaults(func=cmd_sources)

    cal = sub.add_parser("calibrate", help="what threshold the hash can stand on")
    cal.add_argument("--manifest", default="data/processed/manifest.csv")
    cal.add_argument("--out", default="outputs/clinic_hash_calibration.json")
    cal.add_argument("--sizes", default="8,12,16,24")
    cal.add_argument("--n-positive", type=int, default=40)
    cal.add_argument("--n-negative", type=int, default=60)
    cal.add_argument("--seed", type=int, default=0)
    cal.set_defaults(func=cmd_calibrate)

    o = sub.add_parser("overlap", help="pixel-level near-duplicate audit")
    o.add_argument("--manifest", default="data/processed/manifest.csv")
    o.add_argument("--out", default="outputs/clinic_overlap.csv")
    o.add_argument("--threshold", type=int, default=A.DUPLICATE_THRESHOLD,
                   help="max hamming distance; run `calibrate` before changing it")
    o.add_argument("--size", type=int, default=A.HASH_SIZE)
    o.add_argument("--limit", type=int, default=0, help="images per clinic (0 = all)")
    o.add_argument("--show", type=int, default=3, help="example pairs to print")
    o.add_argument("--seed", type=int, default=0)
    o.set_defaults(func=cmd_overlap)

    c = sub.add_parser("confound", help="can capture statistics call the label?")
    c.add_argument("--manifest", default="data/processed/manifest.csv")
    c.add_argument("--out", default="outputs/clinic_confound.csv")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--resize", type=int, default=224,
                   help="also measure on the resampled image the model sees; 0 to skip")
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(func=cmd_confound)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
