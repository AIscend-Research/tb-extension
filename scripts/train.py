#!/usr/bin/env python3
"""Train a single run and save predictions for evaluation.

    # in-distribution reference (random split)
    python scripts/train.py --config configs/base.yaml --mode random

    # one leave-one-cohort-out fold
    python scripts/train.py --config configs/base.yaml --mode loco --fold shenzhen

Writes a checkpoint and a predictions .npz (y_true, prob, uncertainty, cohort_idx)
into --out-dir. Feed the .npz to scripts/evaluate.py with the random-split
accuracy as --acc-id to get the gap-recovery report.

This script needs torch/torchvision/timm and real images (build the manifest
first). For a no-GPU sanity check of the metric pipeline, use
scripts/smoke_test.py instead.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from xctb.utils import load_config, set_seed
from xctb.data.dataset import CXRDataset, cohort_index_map
from xctb.data.transforms import build_transforms
from xctb.data.splits import leave_one_cohort_out, random_split
from xctb.models.model import build_model
from xctb.engine.train import train_one_run
from xctb.engine.infer import predict, ensemble_predict, collect_logits
from xctb.calibration import fit_temperature, apply_temperature
from xctb.eval.metrics import binary_report


def make_loader(df, cfg, train):
    import torch

    ds = CXRDataset(
        df,
        transform=build_transforms(cfg["image_size"], train=train),
        cohort_to_idx=cohort_index_map(),
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=train,
        num_workers=cfg.get("num_workers", 2),
        drop_last=train,
    )


def run_fold(train_df, val_df, test_df, cfg, device, tag, out_dir):
    method = str(cfg.get("uncertainty", "mc_dropout")).lower()
    train_loader = make_loader(train_df, cfg, train=True)
    val_loader = make_loader(val_df, cfg, train=False)
    test_loader = make_loader(test_df, cfg, train=False)

    if method == "ensemble":
        models = []
        for k in range(int(cfg.get("ensemble_size", 3))):
            set_seed(cfg.get("seed", 0) + k)
            m, _ = train_one_run(build_model(cfg), train_loader, val_loader, cfg, device)
            models.append(m)
        pred = ensemble_predict(models, test_loader, device)
        import torch

        torch.save([m.state_dict() for m in models], os.path.join(out_dir, f"{tag}_ensemble.pt"))
    else:
        model, _ = train_one_run(build_model(cfg), train_loader, val_loader, cfg, device)
        # Temperature scaling on the seen-cohort validation split, then applied to test.
        val_logits, val_labels = collect_logits(model, val_loader, device)
        T = fit_temperature(val_logits, val_labels)
        pred = predict(model, test_loader, device, method="mc_dropout", n_samples=cfg.get("mc_samples", 20))
        test_logits, _ = collect_logits(model, test_loader, device)
        pred["prob"] = apply_temperature(test_logits, T)
        import torch

        torch.save({"state_dict": model.state_dict(), "temperature": T}, os.path.join(out_dir, f"{tag}.pt"))
        print(f"  fitted temperature T={T:.3f}")

    out_npz = os.path.join(out_dir, f"{tag}.npz")
    np.savez(out_npz, **pred)
    rep = binary_report(pred["y_true"], pred["prob"])
    print(f"  {tag}: acc={rep['accuracy']:.4f} sens={rep['sensitivity']:.4f} "
          f"spec={rep['specificity']:.4f} auroc={rep['auroc']:.4f}  -> {out_npz}")
    return out_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--mode", choices=["random", "loco"], default="loco")
    ap.add_argument("--fold", default=None, help="loco: run only this held-out cohort")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))
    os.makedirs(args.out_dir, exist_ok=True)

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, config={args.config}, dg={cfg.get('dg_method')}, "
          f"backbone={cfg.get('backbone')}")

    manifest = pd.read_csv(args.manifest)

    if args.mode == "random":
        tr, va, te = random_split(manifest, seed=cfg.get("seed", 0))
        run_fold(tr, va, te, cfg, device, "random", args.out_dir)
    else:
        for tr, va, te, held in leave_one_cohort_out(manifest, seed=cfg.get("seed", 0)):
            if args.fold and held != args.fold:
                continue
            print(f"[fold: hold out {held}]  train={len(tr)} val={len(va)} test={len(te)}")
            run_fold(tr, va, te, cfg, device, f"loco_{held}", args.out_dir)


if __name__ == "__main__":
    main()
