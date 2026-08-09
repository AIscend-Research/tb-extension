#!/usr/bin/env python3
"""Build a severity-sweep eval manifest for the smartphone-degradation study.

Expands an existing manifest into one row per (image, severity), tagged with
the degradation strategy/severity/seed that `xctb.data.dataset.CXRDataset`
needs to reproduce the same degraded image deterministically at load time.
Does not touch pixels or copy any files, so it works before real cohort images
are on disk too (with --synthetic).

    python scripts/build_degraded_eval.py --manifest data/manifest.csv --out data/degraded_manifest.csv
    python scripts/build_degraded_eval.py --synthetic   # no images needed

This is the input for Phase 4's "accuracy at increasing degradation severity"
report: run inference over this manifest and group results by
`degradation_severity`.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from xctb.data.degradation import DEGRADATION_STRATEGIES, build_degradation_manifest
from xctb.data.manifest import synthetic_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="data/degraded_manifest.csv")
    ap.add_argument(
        "--strategy",
        default="full",
        choices=[k for k, v in DEGRADATION_STRATEGIES.items() if v is not None],
    )
    ap.add_argument("--severities", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true", help="use a fake manifest for testing")
    args = ap.parse_args()

    if args.synthetic:
        print("Using a SYNTHETIC manifest (no real images).")
        manifest = synthetic_manifest()
    else:
        if not os.path.isfile(args.manifest):
            sys.exit(
                f"No manifest at {args.manifest}. Run scripts/build_manifest.py first, "
                "or pass --synthetic to test this pipeline without images."
            )
        manifest = pd.read_csv(args.manifest)

    severities = [float(s) for s in args.severities.split(",")]
    out = build_degradation_manifest(manifest, severities=severities, strategy=args.strategy, seed=args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {len(out)} rows ({len(manifest)} images x {len(severities)} severities) to {args.out}\n")
    print(out.groupby(["cohort", "degradation_severity"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
