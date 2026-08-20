#!/usr/bin/env python3
"""Precompute the per-pixel floor for every (image, severity) the arms train on.

The physics cannot be computed inside `__getitem__`: inversion costs ~0.2 s per
photograph, which is longer than a forward-backward pass, and the training
degradation pipeline produces images the certificate cannot read at all (no
fiducials -- see `data/physics_cache.py`). So the captures are made once, through
`physics/film.simulate`, floored, and written to disk.

Every arm in `scripts/measure_physics_in_training.py` -- including the baseline
and the scrambled control -- reads the *same* cache. That is what makes the
comparison a comparison: the only thing that differs between arms is what the
network is allowed to do with the floor, never which photographs it saw.

    python scripts/build_physics_cache.py --out data/processed/physics_cache
    python scripts/build_physics_cache.py --limit 40 --workers 8      # a smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from tbtrust.data import physics_cache as PC

_W: dict = {}


def _init(size: int, out_size: int, seed: int, outdir: str):
    _W.update(size=size, out_size=out_size, seed=seed, outdir=Path(outdir))


def _one(job):
    path, severity = job
    out = _W["outdir"] / f"{PC.cache_key(path, severity)}.npz"
    if out.exists():
        return {"key": out.stem, "cached": True}
    item = PC.compute_one(path, severity, size=_W["size"], out_size=_W["out_size"],
                          seed=_W["seed"])
    PC.save(item, out)
    return {"key": out.stem, "cached": False, **{k: item.meta.get(k) for k in
            ("verdict", "margin_db", "limiting_factor", "abstained")}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--out", default="data/processed/physics_cache")
    ap.add_argument("--size", type=int, default=512,
                    help="capture resolution the inversion runs at")
    ap.add_argument("--out-size", type=int, default=224,
                    help="stored resolution; must match data.image_size")
    ap.add_argument("--severities", default=",".join(str(s) for s in PC.SEVERITIES))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if args.limit:
        df = df.head(args.limit)
    sevs = [float(x) for x in args.severities.split(",")]
    jobs = [(str(p), s) for p in df["path"] for s in sevs]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.size, args.out_size, args.seed,
                                       str(outdir))) as ex:
        for i, r in enumerate(ex.map(_one, jobs, chunksize=4)):
            rows.append(r)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)

    fresh = [r for r in rows if not r.get("cached")]
    abstain = [r for r in fresh if r.get("abstained")]
    index = {
        "manifest": args.manifest, "size": args.size, "out_size": args.out_size,
        "severities": sevs, "seed": args.seed, "n_items": len(rows),
        "n_computed": len(fresh),
        "abstain_rate": (len(abstain) / len(fresh)) if fresh else None,
    }
    (outdir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"{len(rows)} items in {outdir} "
          f"({len(fresh)} computed, abstain rate "
          f"{index['abstain_rate'] if index['abstain_rate'] is None else round(index['abstain_rate'], 3)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
