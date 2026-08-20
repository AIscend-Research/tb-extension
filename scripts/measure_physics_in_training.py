#!/usr/bin/env python3
"""Does measured capture quality help the classifier *learn*, or only abstain?

The certificate is currently a post-hoc gate: `eval/physics_deferral.py` ranks a
trained model's predictions by whether the photograph could carry the finding.
That leaves the obvious question untested -- if the physics knows something about
the image, does handing it to the network during training make the network
better, rather than only telling us when to distrust it?

Six arms, one cache, one seed set. Every arm trains on the *same* precomputed
captures at the same severities; they differ only in what the network is allowed
to do with the floor:

    control    photograph only. The comparison baseline.
    channel    normalised per-pixel floor as a fourth input channel.
    scramble   a fourth channel holding *another* image's floor map.
    severity   a fourth channel of one constant: the applied severity.
    weight_dn  loss down-weighted where the certificate says the photograph
               cannot carry the finding.
    weight_up  the same weighting, inverted.

The two control arms are the point of the design, and without them a positive
result would be uninterpretable:

* **scramble** has the identical marginal distribution to `channel` and none of
  the pairing. A smooth extra channel can act as a regulariser and buy accuracy
  while carrying nothing about the image it is attached to. If `channel` beats
  `control` and `scramble` beats it by the same margin, the physics contributed
  nothing.
* **severity** is a single scalar the simulator already knew. If it matches
  `channel`, the per-pixel measurement -- the expensive, novel part -- is not
  what is paying, and the honest write-up is "capture severity helps, and you do
  not need a certificate to get it".

Reported per arm over `--seeds` repetitions: held-out-clinic accuracy, AUC and
sensitivity, with the paired difference against `control` and a bootstrap
interval on it. Paired because the arms share seeds, which removes the run-to-run
variance that would otherwise swamp any effect this small corpus can show.

    python scripts/measure_physics_in_training.py --seeds 3
    python scripts/measure_physics_in_training.py --arms control channel scramble
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

ARMS = {
    "control":   {"mode": "none",     "loss_weight": "none"},
    "channel":   {"mode": "channel",  "loss_weight": "none"},
    "scramble":  {"mode": "scramble", "loss_weight": "none"},
    "severity":  {"mode": "severity", "loss_weight": "none"},
    "weight_dn": {"mode": "none",     "loss_weight": "down"},
    "weight_up": {"mode": "none",     "loss_weight": "up"},
}


def _test_metrics(cfg: dict, checkpoint: Path, severities) -> dict:
    """Held-out-clinic metrics, at each cached severity and pooled.

    The test clinic is scored through the same cached captures the arms trained
    on, for the same reason the arms share a cache: a different capture draw at
    test time would put noise between the arms that has nothing to do with what
    is being compared.
    """
    import torch
    from torch.utils.data import DataLoader

    from tbtrust.data import manifest as M
    from tbtrust.data.dataset import PhysicsCachedDataset
    from tbtrust.data.physics_cache import CacheStats
    from tbtrust.data.splits import split_from_config
    from tbtrust.eval.metrics import accuracy, roc_auc, sensitivity, specificity
    from tbtrust.models.baseline import build_model
    from tbtrust.utils.io import load_checkpoint, pick_device

    df = split_from_config(M.load(cfg["data"]["manifest"]), cfg)
    cache = Path(cfg["physics"]["cache"])
    stats = None
    if (cache / "stats.json").exists():
        stats = CacheStats.from_dict(json.loads((cache / "stats.json").read_text()))

    device = pick_device(cfg["train"].get("device"))
    model = build_model(cfg).to(device)
    load_checkpoint(model, checkpoint, map_location=device)
    model.eval()

    rows, ys, ps = [], [], []
    for sev in severities:
        ds = PhysicsCachedDataset(
            df, cache_dir=cache, split="test", severities=(sev,),
            physics_mode=cfg["physics"]["mode"], stats=stats, seed=cfg.get("seed", 0),
            epoch_severity=False)
        if len(ds) == 0:
            continue
        y_all, p_all = [], []
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=32, num_workers=2):
                p = torch.sigmoid(model(batch["image"].to(device))["logit"])
                p_all.append(p.cpu().numpy())
                y_all.append(batch["label"].numpy())
        y = np.concatenate(y_all)
        p = np.concatenate(p_all)
        ys.append(y)
        ps.append(p)
        rows.append({"severity": sev, "accuracy": accuracy(y, p), "auc": roc_auc(y, p),
                     "sensitivity": sensitivity(y, p), "specificity": specificity(y, p),
                     "n": len(y)})
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return {"pooled": {"accuracy": accuracy(y, p), "auc": roc_auc(y, p),
                       "sensitivity": sensitivity(y, p), "specificity": specificity(y, p),
                       "n": len(y)},
            "by_severity": rows}


def run_arm(name: str, arm: dict, seed: int, args) -> dict:
    from tbtrust.config import load_experiment
    from tbtrust.train.loop import train

    cfg = load_experiment(args.config)
    cfg["seed"] = seed
    cfg["physics"]["cache"] = args.cache
    cfg["physics"]["mode"] = arm["mode"]
    cfg["physics"]["loss_weight"] = arm["loss_weight"]
    cfg["train"]["epochs"] = args.epochs
    cfg["train"]["output_dir"] = str(Path(args.outdir) / f"{name}_s{seed}")
    if args.device:
        cfg["train"]["device"] = args.device

    t0 = time.time()
    done = _finished_run(cfg)
    if args.resume and done is not None:
        # Training is ~7x the cost of scoring, and an eighteen-run sweep that
        # loses everything to one interruption is a design fault rather than bad
        # luck. `train_summary.json` is the completion marker because `train()`
        # writes it only after the last epoch -- a `best.ckpt` on its own can be
        # the best of three epochs of a run that was killed at four, and reusing
        # that would silently put a shorter-trained arm into the comparison.
        summary, checkpoint = done
        cfg.setdefault("model", {})["in_channels"] = 3 if arm["mode"] == "none" else 4
        res = {"best_val_accuracy": summary["best_val_accuracy"],
               "checkpoint": str(checkpoint)}
        resumed = True
    else:
        res = train(cfg)
        resumed = False

    metrics = _test_metrics(cfg, Path(res["checkpoint"]),
                            [float(s) for s in args.severities.split(",")])
    return {"arm": name, "seed": seed, "best_val_accuracy": res["best_val_accuracy"],
            "resumed": resumed, "seconds": time.time() - t0,
            **{f"test_{k}": v for k, v in metrics["pooled"].items()},
            "by_severity": metrics["by_severity"]}


def _finished_run(cfg: dict):
    """(summary, checkpoint) for a run that reached its last epoch, else None."""
    out = Path(cfg["train"]["output_dir"]) / cfg["data"]["holdout_clinic"]
    summary, ckpt = out / "train_summary.json", out / "best.ckpt"
    if not (summary.exists() and ckpt.exists()):
        return None
    return json.loads(summary.read_text()), ckpt


def paired_summary(df: pd.DataFrame, metric: str, n_boot: int = 5000,
                   seed: int = 0, reference: str = "control") -> list[dict]:
    """Per-arm mean and its paired difference against `reference`, with a CI.

    Paired over seeds: arms share initialisation and data order, so the
    difference has far less variance than either arm's own spread. Reporting the
    unpaired difference on a corpus this size would drown any real effect in
    run-to-run noise, and reporting no interval at all would let a two-run fluke
    read as a finding.

    `reference` exists because two different questions are being asked and only
    one of them is about physics. Against `control`, a difference says a fourth
    input channel changed something. Against `scramble` -- which holds the extra
    channel and destroys only its pairing to the image -- a difference says
    *this photograph's measured floor* changed something. Only the second is a
    claim about the certificate.
    """
    rng = np.random.default_rng(seed)
    ref = df[df["arm"] == reference].set_index("seed")[metric]
    out = []
    for arm in df["arm"].unique():
        a = df[df["arm"] == arm].set_index("seed")[metric]
        shared = sorted(set(a.index) & set(ref.index))
        d = (a.loc[shared] - ref.loc[shared]).to_numpy(float)
        row = {"arm": arm, "metric": metric, "reference": reference,
               "n_seeds": len(a),
               "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else float("nan"),
               "delta_vs_reference": float(d.mean()) if len(d) else float("nan")}
        if len(d) > 1:
            boot = [float(rng.choice(d, size=len(d), replace=True).mean())
                    for _ in range(n_boot)]
            row["delta_lo"], row["delta_hi"] = (float(np.quantile(boot, 0.025)),
                                               float(np.quantile(boot, 0.975)))
            row["separates"] = bool(row["delta_lo"] > 0 or row["delta_hi"] < 0)
        else:
            row["delta_lo"] = row["delta_hi"] = float("nan")
            row["separates"] = False
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/physics_train_montgomery.yaml")
    ap.add_argument("--cache", default="data/processed/physics_cache")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--severities", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--outdir", default="outputs/physics_training")
    ap.add_argument("--out", default="outputs/physics_in_training.csv")
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="reuse runs that already reached their last epoch")
    args = ap.parse_args()

    unknown = [a for a in args.arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {list(ARMS)}")
    if "control" not in args.arms:
        raise SystemExit("the control arm is the comparison; it cannot be dropped")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.stem + "_partial.csv")
    rows = []
    for seed in range(args.seeds):
        for name in args.arms:
            print(f"\n=== {name} seed {seed} ===", flush=True)
            rows.append(run_arm(name, ARMS[name], seed, args))
            # Flushed every run, because the first version wrote the CSV only at
            # the end and an interruption at run 16 of 18 left nothing on disk.
            # To a *sibling* path, not to `out`: writing partial results over
            # `out` destroys a complete previous sweep from the first run of a
            # resumed one, which is exactly what happened the first time this
            # was tried. `out` is written once, at the end, when it is complete.
            pd.DataFrame([{k: v for k, v in r.items() if k != "by_severity"}
                          for r in rows]).to_csv(partial, index=False)
            r = rows[-1]
            tag = " [resumed]" if r["resumed"] else ""
            print(f"  val={r['best_val_accuracy']:.4f} test_acc={r['test_accuracy']:.4f} "
                  f"test_auc={r['test_auc']:.4f} ({r['seconds']:.0f}s){tag}", flush=True)

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "by_severity"} for r in rows])
    df.to_csv(out, index=False)
    partial.unlink(missing_ok=True)

    summary = []
    for metric in ("test_accuracy", "test_auc", "test_sensitivity"):
        summary += paired_summary(df, metric)
        # The contrast the design exists for. `channel` against `control` only
        # says a fourth channel changed something; `channel` against `scramble`
        # says whether *this image's* measured floor did, since scramble holds
        # the extra channel fixed and destroys only its pairing. A physics claim
        # rests on this comparison, not on the one against control.
        if {"channel", "scramble"} <= set(df["arm"]):
            summary += [r for r in paired_summary(df, metric, reference="scramble")
                        if r["arm"] in {"channel", "severity"}]
    pd.DataFrame(summary).to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    Path(out.with_suffix(".json")).write_text(json.dumps(rows, indent=2, default=float))

    for ref in ("control", "scramble"):
        block = [r for r in summary if r["reference"] == ref and r["arm"] != ref]
        if not block:
            continue
        print(f"\npaired difference against {ref} (positive = arm is better)")
        for r in block:
            star = "  *" if r["separates"] else ""
            print(f"  {r['metric']:18s} {r['arm']:10s} "
                  f"{r['delta_vs_reference']:+.4f} "
                  f"[{r['delta_lo']:+.4f}, {r['delta_hi']:+.4f}]{star}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
