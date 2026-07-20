#!/usr/bin/env python3
"""Score a saved predictions file. Torch-free.

A predictions file is a .npz with arrays y_true, prob, uncertainty (and
optionally cohort_idx), written by scripts/train.py. Given the in-distribution
reference accuracy (from the random-split run), this prints the headline numbers:
the generalization gap, the risk-coverage / AURC, calibration error, and how
much of the gap deferral recovers.

    python scripts/evaluate.py --pred runs/loco_shenzhen.npz --acc-id 0.97
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from xctb.eval.metrics import binary_report
from xctb.eval.deferral import (
    risk_coverage_curve,
    generalization_gap_recovery,
    coverage_to_recover,
)
from xctb.calibration import expected_calibration_error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help=".npz with y_true, prob, uncertainty")
    ap.add_argument("--acc-id", type=float, required=True, help="in-distribution (random-split) accuracy")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.pred)
    y_true, prob, unc = d["y_true"], d["prob"], d["uncertainty"]

    report = binary_report(y_true, prob, args.threshold)
    coverage, risk, aurc = risk_coverage_curve(y_true, prob, unc, args.threshold)
    recovery = generalization_gap_recovery(args.acc_id, y_true, prob, unc, threshold=args.threshold)
    reach90 = coverage_to_recover(args.acc_id, y_true, prob, unc, target_fraction=0.9, threshold=args.threshold)
    ece = expected_calibration_error(prob, y_true)

    result = {
        "classification": report,
        "aurc": round(float(aurc), 4),
        "ece": round(float(ece), 4),
        "gap_recovery": recovery,
        "coverage_to_recover_90pct": (
            {"coverage": round(reach90[0], 3), "deferred": round(reach90[1], 3), "accuracy": round(reach90[2], 4)}
            if reach90
            else None
        ),
    }
    print(json.dumps(result, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
