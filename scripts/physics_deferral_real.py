#!/usr/bin/env python3
"""Join a trained checkpoint's calibrated probabilities to physics certificates.

`notebooks/08_physics_deferral_and_triage.ipynb` fits a stand-in logistic
regression so it runs standalone with no checkpoint and no GPU. This script is
the real version of that cell: it produces one row per (image, severity) with

    label, prob (temperature-scaled), mc_std, margin_db, abstained, triage_*

which is exactly the table `eval/physics_deferral.py` consumes. Point the
notebook at the CSV with TBTRUST_PROBS and every number below it becomes a real
one.

Two things this does that a naive join cannot, and both are the reason it exists
rather than being a merge of `outputs/certs_*.csv` with a separate eval run:

1. **One photograph, two readings.** The certificate and the classifier see the
   *same* simulated capture array, because both come off one `film.simulate`
   call per row. Joining an archived certificate table to a separate model pass
   would pair a margin measured on one draw of the capture noise with a
   probability measured on another; the complementarity claim -- that the
   physics catches errors the learned signal misses -- is about individual
   photographs, so that pairing has to be exact.

2. **The protocol from `eval/run.py`, unchanged.** The temperature, the
   MC-dropout confidence map and the deferral threshold are all fitted on the
   *validation* split at the config's primary severity and then applied, never
   re-fitted, to the held-out test rows. The physics gate needs no fitting at
   all -- margin <= 0 dB is the certificate's own rule -- which is a large part
   of why it is worth comparing against.

Only the test-split images are certified (the expensive step, ~1.6 s each at
1024 px), fanned out over a process pool. The validation pass needs the photo
but not the certificate, so it is cheap.

    python scripts/physics_deferral_real.py \
        --checkpoint outputs/rnd_clean_s0/montgomery/best.ckpt \
        --out outputs/physics_deferral_real.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _photo(path: str, severity: float, size: int, seed: int) -> np.ndarray:
    """Re-photograph one archive image through the forward capture model."""
    from tbtrust.physics.film import simulate
    from tbtrust.utils.seed import capture_seed

    img = Image.open(path).convert("L")
    if max(img.size) != size:
        img = img.resize((size, size), Image.BILINEAR)
    photo, _ = simulate(np.asarray(img), severity=float(severity),
                        rng=np.random.default_rng(capture_seed(path, severity, seed)),
                        size=size)
    return photo


def _to_model_input(photo: np.ndarray, image_size: int) -> np.ndarray:
    """Photo -> the exact tensor layout TBDataset feeds the network."""
    arr = np.asarray(Image.fromarray(np.asarray(photo, dtype=np.uint8))
                     .resize((image_size, image_size), Image.BILINEAR), dtype=np.float32) / 255.0
    arr = np.stack([arr, arr, arr], axis=0)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    return (arr - mean) / std


# Pool workers get their configuration through module globals rather than through
# every task tuple: the findings table and the floor spec are identical for all
# rows and pickling them per row is pure overhead.
_W: dict = {}


def _worker_init(size: int, image_size: int, seed: int, rose_k: float):
    from tbtrust.physics import findings as FIND
    from tbtrust.physics.floor import FloorSpec

    _W.update(size=size, image_size=image_size, seed=seed,
              findings=FIND.core(), spec=FloorSpec(rose_k=rose_k))


def _certify_one(task):
    """One (path, severity): photograph it once, then certify *and* score that photo."""
    from tbtrust.physics.certificate import certificate_confidence, certify
    from tbtrust.physics.invert import invert
    from tbtrust.physics.triage import triage

    path, severity = task
    photo = _photo(path, severity, _W["size"], _W["seed"])
    cal = invert(photo)
    cert = certify(cal, findings=_W["findings"], spec=_W["spec"])
    dec = triage(cert, cal)                     # confidence-free; the tie is broken below
    row = {
        "path": path,
        "severity": float(severity),
        **cert.as_dict(),
        "physics_confidence": certificate_confidence(cert),
        "abstained": cert.abstained,
        "triage_action": dec.action.value,
        "triage_reason": dec.reason,
        "triage_instruction": dec.instruction,
        "expected_retake_gain_db": dec.expected_gain_db,
        **{f"cal_{k}": v for k, v in cal.summary().items()},
    }
    return row, _to_model_input(photo, _W["image_size"])


def _collect(model, batch: np.ndarray, device: str, mc_passes: int):
    """Deterministic logits plus MC-dropout spread, on one shared batch tensor.

    Same rule as `eval/run.py._collect`: the stochastic passes reuse the very
    tensor the deterministic pass saw, so `mc_std[i]` describes the same
    photograph as `logit[i]`.
    """
    import torch

    from tbtrust.models.uncertainty import enable_mc_dropout

    x = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        model.eval()
        logit = model(x)["logit"].cpu().numpy()
        mc = None
        if mc_passes > 0:
            enable_mc_dropout(model)
            samples = torch.stack([torch.sigmoid(model(x)["logit"]) for _ in range(mc_passes)])
            mc = samples.std(dim=0).cpu().numpy()
        model.eval()
    return logit, mc


def _score_photos(model, photos, device: str, mc_passes: int, batch_size: int):
    logits, stds = [], []
    for i in range(0, len(photos), batch_size):
        chunk = np.stack(photos[i:i + batch_size])
        lg, mc = _collect(model, chunk, device, mc_passes)
        logits.append(lg)
        if mc is not None:
            stds.append(mc)
    return (np.concatenate(logits),
            np.concatenate(stds) if stds else None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None,
                    help="config YAML; defaults to the one stored in the checkpoint")
    ap.add_argument("--config-dir", default="configs")
    ap.add_argument("--certs", default=None,
                    help="certificate table naming which archive images to use "
                         "(only its `path` column is read; the certificates are recomputed "
                         "so photo and probability come off one capture)")
    ap.add_argument("--out", default="outputs/physics_deferral_real.csv")
    ap.add_argument("--size", type=int, default=1024, help="physics working resolution")
    ap.add_argument("--severities", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--limit", type=int, default=None, help="cap on test images (debugging)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from tbtrust.data import manifest as M
    from tbtrust.data.splits import split_from_config
    from tbtrust.eval import calibration as C
    from tbtrust.eval import deferral as D
    from tbtrust.eval.run import _build_model
    from tbtrust.utils.io import load_checkpoint
    from tbtrust.utils.seed import seed_everything

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.config:
        from tbtrust.config import load_config

        cfg = load_config(args.config, config_dir=args.config_dir)
    else:
        cfg = ckpt["config"]
    eval_cfg = cfg.get("eval", {})
    seed_everything(int(eval_cfg.get("seed", cfg.get("seed", 0))))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(cfg).to(device)
    load_checkpoint(model, args.checkpoint, map_location=device)
    image_size = int(cfg["data"]["image_size"])
    mc_passes = int(eval_cfg.get("mc_dropout_passes", 20))
    batch_size = int(cfg["train"]["batch_size"])
    primary = float(eval_cfg.get("primary_severity", 0.0))

    # The model's own split. Scoring images the network trained on would make
    # every number below a memorisation measurement.
    df = split_from_config(M.load(cfg["data"]["manifest"]), cfg)

    # The manifest the model was trained from is a cache of resized copies; the
    # certificates are computed from the full-resolution archive files. Match
    # them on the archive filename, which survives both naming schemes.
    def key(p: str) -> str:
        return os.path.basename(p).split("__")[-1]

    df["key"] = df["path"].map(key)
    raw = M.load(cfg["data"].get("raw_manifest", "data/processed/manifest.csv"))
    raw["key"] = raw["path"].map(key)
    raw_path = dict(zip(raw["key"], raw["path"], strict=False))

    wanted = None
    if args.certs:
        wanted = {key(p) for p in pd.read_csv(args.certs)["path"].unique()}

    def rows_for(split: str) -> pd.DataFrame:
        d = df[df["split"] == split].copy()
        d = d[d["key"].isin(raw_path)]
        if wanted is not None:
            d = d[d["key"].isin(wanted)]
        d["raw"] = d["key"].map(raw_path)
        return d.reset_index(drop=True)

    val, test = rows_for("val"), rows_for("test")
    if args.limit:
        test = test.head(args.limit)
    if val.empty or test["label"].nunique() < 2:
        print("need a non-empty val split and a two-class test split; got "
              f"{len(val)} val / {len(test)} test rows", file=sys.stderr)
        return 1
    severities = [float(s) for s in args.severities.split(",") if s.strip()]
    print(f"val {len(val)} images, test {len(test)} images x {len(severities)} severities "
          f"= {len(test) * len(severities)} certificates", flush=True)

    # ------------------------------------------------------------------ val fits
    # The certificate is not needed here, only the photograph, so this pass skips
    # the inversion entirely and runs in seconds.
    val_photos = [_to_model_input(_photo(p, primary, args.size, args.seed), image_size)
                  for p in val["raw"]]
    v_logit, v_mc = _score_photos(model, val_photos, device, mc_passes, batch_size)
    y_val = val["label"].to_numpy().astype(int)

    bounds = tuple(eval_cfg.get("temperature_bounds", (0.05, 20.0)))
    temperature = C.fit_temperature(v_logit, y_val, bounds=bounds)
    p_val = C.apply_temperature(v_logit, temperature)
    at_bound = min(abs(temperature - bounds[0]), abs(temperature - bounds[1])) < 1e-3
    print(f"temperature {temperature:.4g}" + ("  [ON SEARCH BOUND -- treat as uncalibrated]"
                                              if at_bound else ""))
    print(f"val ECE {C.expected_calibration_error(y_val, C.apply_temperature(v_logit, 1.0)):.3f}"
          f" -> {C.expected_calibration_error(y_val, p_val):.3f} after temperature")

    tuned = {}
    lo, hi = (float(np.min(v_mc)), float(np.max(v_mc))) if v_mc is not None else (0.0, 0.0)

    def mc_conf(u):
        """The val-fitted MC-dropout -> confidence map, frozen for the test split."""
        u = np.asarray(u, dtype=float)
        if hi - lo < 1e-12:
            return np.full_like(u, 0.5)
        return np.clip(1.0 - (u - lo) / (hi - lo), 0.0, 1.0)

    for method, conf_val in (("confidence", None),
                             ("mc_dropout", mc_conf(v_mc) if v_mc is not None else None)):
        if method != "confidence" and conf_val is None:
            continue
        pt = D.tune_threshold(y_val, p_val,
                              target=eval_cfg.get("target", "accuracy"),
                              target_value=eval_cfg.get("target_accuracy", 0.99),
                              min_coverage=eval_cfg.get("min_coverage", 0.5),
                              confidence=conf_val)
        tuned[method] = pt.threshold
        print(f"threshold[{method}] = {pt.threshold:.3f} "
              f"(val coverage {pt.coverage:.2f}, val accuracy {pt.accuracy:.2f})")

    # -------------------------------------------------------- test certificates
    from multiprocessing import get_context

    tasks = [(r.raw, s) for r in test.itertuples() for s in severities]
    ctx = get_context("spawn")
    rows, photos = [], []
    with ctx.Pool(args.workers, initializer=_worker_init,
                  initargs=(args.size, image_size, args.seed,
                            float(eval_cfg.get("rose_k", 5.0)))) as pool:
        for i, (row, photo) in enumerate(pool.imap(_certify_one, tasks, chunksize=1), start=1):
            rows.append(row)
            photos.append(photo)
            if i % 25 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} certified", flush=True)

    out = pd.DataFrame(rows)
    meta = test.set_index("raw")[["key", "clinic", "label"]]
    out = out.join(meta, on="path")

    t_logit, t_mc = _score_photos(model, photos, device, mc_passes, batch_size)
    out["logit"] = t_logit
    out["prob"] = C.apply_temperature(t_logit, temperature)
    out["mc_std"] = t_mc if t_mc is not None else np.nan
    out["learned_confidence"] = np.maximum(out["prob"], 1 - out["prob"])
    out["mc_confidence"] = mc_conf(out["mc_std"]) if t_mc is not None else np.nan
    out["temperature"] = temperature
    out["threshold_confidence"] = tuned.get("confidence", np.nan)
    out["threshold_mc_dropout"] = tuned.get("mc_dropout", np.nan)
    out["is_test"] = True

    # `triage` only consults the model where the capture is adequate: an image the
    # certificate clears but the classifier is unsure about is a REFER, because a
    # retake reproduces the same photograph. The worker had no probability yet, so
    # apply that one branch here rather than re-running the inversion for it.
    # The cutoff is the threshold tuned on val, not 0.5: max(p, 1-p) is >= 0.5 by
    # construction, so comparing against 0.5 would make "the classifier is unsure"
    # unreachable and silently delete the REFER path.
    cut = tuned.get("confidence", 0.5)
    unsure = (out["triage_action"] == "report") & (out["learned_confidence"] < cut)
    out.loc[unsure, ["triage_action", "triage_reason", "triage_instruction"]] = [
        "refer", "model_uncertain_adequate_capture",
        ("Image quality is adequate -- the photograph carries the density detail a screening "
         "read needs -- but the finding is ambiguous. A retake will not help. Refer for "
         "specialist review."),
    ]
    out["model_confident"] = out["learned_confidence"] >= 0.5

    # Control: the same held-out images as the network was trained to see them --
    # the cached 224 px archive copies, no film, no re-photography. It separates
    # "this checkpoint is weak" from "the capture shift is what breaks it", and
    # without it a low accuracy on the simulated captures is unattributable.
    clean = [_to_model_input(np.asarray(Image.open(p_).convert("L")), image_size)
             for p_ in test["path"]]
    c_logit, _ = _score_photos(model, clean, device, 0, batch_size)
    clean_acc = float(((C.apply_temperature(c_logit, temperature) >= 0.5).astype(int)
                       == test["label"].to_numpy().astype(int)).mean())
    out["clean_test_accuracy"] = clean_acc
    print(f"control -- same images, no re-photography: accuracy {clean_acc:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out)} rows, {out['path'].nunique()} images)")

    acc = float(((out["prob"] >= 0.5).astype(int) == out["label"]).mean())
    print(f"test accuracy across the sweep: {acc:.3f}   "
          f"retake {float((out['triage_action'] == 'retake').mean()):.2f} / "
          f"refer {float((out['triage_action'] == 'refer').mean()):.2f} / "
          f"report {float((out['triage_action'] == 'report').mean()):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
