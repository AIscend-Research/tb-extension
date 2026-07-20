#!/usr/bin/env python3
"""Report per-cohort image statistics (brightness, contrast, resolution).

    python scripts/cohort_stats.py --manifest data/manifest.csv --sample 100

Gives you the concrete "how different are these machines" numbers for the paper.
Needs the real images on disk (does nothing useful on a synthetic manifest).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from xctb.data.cohort_stats import cohort_shift_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--out", default="data/cohort_shift.csv")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    table = cohort_shift_table(manifest, sample_per_cohort=args.sample)
    print(table.to_string(index=False))
    table.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
