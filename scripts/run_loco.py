#!/usr/bin/env python3
"""Run the whole experiment: the in-distribution reference plus every
leave-one-cohort-out fold, then aggregate.

    python scripts/run_loco.py --config configs/base.yaml

Steps, in order:
  1. random-split run  -> the optimistic in-distribution accuracy (acc_id)
  2. one loco fold per cohort, holding each out in turn
  3. for each fold, compute the gap vs acc_id and the recovery from deferral
  4. print and save a summary table across folds

Compare two configs (for example dg_method: none vs coral) by running this
twice with different config files and diffing the summaries. That comparison is
the paper's main result.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from xctb.utils import load_config, set_seed
from xctb.data.splits import leave_one_cohort_out, random_split
from xctb.eval.metrics import binary_report
from xctb.eval.deferral import generalization_gap_recovery, coverage_to_recover, risk_coverage_curve
from xctb.calibration import expected_calibration_error

# reuse the single-run machinery from scripts/train.py
import scripts.train as train_mod  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))
    os.makedirs(args.out_dir, exist_ok=True)

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(args.manifest)

    # 1. in-distribution reference
    print("=== in-distribution reference (random split) ===")
    tr, va, te = random_split(manifest, seed=cfg.get("seed", 0))
    ref_npz = train_mod.run_fold(tr, va, te, cfg, device, "random", args.out_dir)
    ref = np.load(ref_npz)
    acc_id = binary_report(ref["y_true"], ref["prob"])["accuracy"]
    print(f"acc_id = {acc_id:.4f}\n")

    # 2 + 3. per-fold
    rows = []
    for tr, va, te, held in leave_one_cohort_out(manifest, seed=cfg.get("seed", 0)):
        print(f"=== hold out {held} (train={len(tr)} val={len(va)} test={len(te)}) ===")
        npz_path = train_mod.run_fold(tr, va, te, cfg, device, f"loco_{held}", args.out_dir)
        d = np.load(npz_path)
        rep = binary_report(d["y_true"], d["prob"])
        _, _, aurc = risk_coverage_curve(d["y_true"], d["prob"], d["uncertainty"])
        rec = generalization_gap_recovery(acc_id, d["y_true"], d["prob"], d["uncertainty"])
        reach90 = coverage_to_recover(acc_id, d["y_true"], d["prob"], d["uncertainty"], 0.9)
        rows.append(
            {
                "held_out": held,
                "acc_cross_cohort": rep["accuracy"],
                "sensitivity": rep["sensitivity"],
                "specificity": rep["specificity"],
                "auroc": rep["auroc"],
                "gap": rec["gap"],
                "aurc": round(float(aurc), 4),
                "ece": round(expected_calibration_error(d["prob"], d["y_true"]), 4),
                "deferral_to_recover_90pct": (round(reach90[1], 3) if reach90 else None),
            }
        )

    summary = pd.DataFrame(rows)
    print("\n=== summary across folds ===")
    print(summary.to_string(index=False))
    out_csv = os.path.join(args.out_dir, "loco_summary.csv")
    summary.to_csv(out_csv, index=False)
    with open(os.path.join(args.out_dir, "loco_summary.json"), "w") as f:
        json.dump({"acc_id": acc_id, "config": args.config, "folds": rows}, f, indent=2)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
