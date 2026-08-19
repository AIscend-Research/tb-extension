#!/usr/bin/env python3
"""Score the blind inversion against real phone captures of the printed phantom.

This is the open-loop counterpart to `scripts/validate_physics.py`. That script
scores the estimator against `physics/film.py`, our own forward model, which can
only ever show that the estimator recovers the parameters we handed it. This one
puts a real phone, a real lens, real veiling glare and a real JPEG encoder in the
path, and a calibrated step wedge on the other side of them.

Two stages, and the split is the design (see `physics/recapture.py`):

    reference captures  --wedge-->  what the print actually carries
    test captures       --blind-->  what the estimator says it carries

The manifest says which frames are which. Reference frames are the best captures
the operator could manage; test frames are the experiment -- angles, distances,
room light, phones.

    # dry run: no printer, no phone, forward-model captures through the same path
    python scripts/validate_real_recapture.py --phantom outputs/phantom --dry-run

    # the real thing
    python scripts/validate_real_recapture.py --phantom outputs/phantom \\
        --manifest data/real_recapture/manifest.csv --out outputs/real_recapture

Exit code is 1 if the run fails its own pre-registered gates (see
docs/REAL_RECAPTURE.md), so it can be wired into CI once data exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from PIL import Image

from tbtrust.physics import phantom as PH
from tbtrust.physics import recapture as RC

# Pre-registered gates. Written down before the data existed, which is the only
# time thresholds can be set honestly -- see docs/REAL_RECAPTURE.md for what each
# one is defending against and why it sits where it does.
GATES = {
    # After removing a constant offset. The offset is the systematic term
    # invert.py already admits it cannot pin; what is left is the part that
    # survives differencing, and the floor depends on it directly.
    "tone_rmse_after_offset_od": 0.05,
    # The estimator's PSF against the phantom's interior edge, as a ratio.
    "blur_ratio_band": (0.5, 2.0),
    # Discs the certificate cleared *with margin* that no matched filter came
    # close to finding. The asymmetric failure: in a clinic this is an image
    # passed as adequate that could not carry the finding. Gated on the clear
    # subset rather than the raw count, because a disc a hair above the floor and
    # a hair below d' = k is the bound being approximately right at the hardest
    # point, not a falsification -- and a gate that fires on those would fail on
    # the forward model too, where there is nothing to catch.
    "max_clear_violation_rate": 0.05,
    # The raw rate is reported and not gated; it is the number to watch drift.
    "report_unsafe_rate": True,
    # Reference captures must agree with each other before anything is compared
    # against them.
    "max_reference_spread_od": 0.03,
}


def _load(path: str) -> np.ndarray | None:
    try:
        img = Image.open(path)
    except (OSError, ValueError):
        return None
    return np.asarray(img.convert("L"))


def _phantom_from(dir_path: Path) -> PH.Phantom:
    build = json.loads((dir_path / "phantom_build.json").read_text())["build"]
    return PH.build_from(build)


def _dry_run_captures(ph: PH.Phantom, n_ref: int, n_test: int):
    """Forward-model captures, so the whole path runs with no data.

    Every number this produces is a closed loop and none of it is evidence about
    a real phone. It is here to exercise the analysis and to let the gates be
    sanity-checked before anyone prints anything.
    """
    refs = [RC.simulate_capture(ph, severity=0.0, rng=np.random.default_rng(100 + i))[0]
            for i in range(n_ref)]
    tests = []
    for i in range(n_test):
        sev = round(0.15 + 0.5 * i / max(n_test - 1, 1), 3)
        photo, _ = RC.simulate_capture(ph, severity=sev, rng=np.random.default_rng(200 + i))
        tests.append((photo, {"capture_path": f"<dry-run sev {sev}>", "phone_model": "forward_model",
                              "condition": f"severity_{sev}", "severity": sev}))
    return refs, tests


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phantom", required=True,
                    help="directory written by scripts/make_phantom_film.py")
    ap.add_argument("--manifest", default=None,
                    help="capture log; see data/real_recapture/manifest_template.csv")
    ap.add_argument("--out", default="outputs/real_recapture")
    ap.add_argument("--dry-run", action="store_true",
                    help="synthesise captures through the forward model instead of reading files. "
                         "Exercises the analysis; proves nothing about a real phone")
    ap.add_argument("--dry-run-n", type=int, default=4, help="test captures to synthesise")
    ap.add_argument("--rose-k", type=float, default=5.0,
                    help="d' a disc must reach to count as empirically detectable")
    args = ap.parse_args()

    ph_dir = Path(args.phantom)
    if not (ph_dir / "phantom_build.json").exists():
        print(f"{ph_dir}/phantom_build.json not found. Generate the sheet first:\n"
              f"    python scripts/make_phantom_film.py --out {ph_dir}", file=sys.stderr)
        return 1
    ph = _phantom_from(ph_dir)
    from tbtrust.physics.floor import FloorSpec

    spec = FloorSpec(rose_k=args.rose_k)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- the captures
    if args.dry_run:
        refs, tests = _dry_run_captures(ph, 3, args.dry_run_n)
        print(f"DRY RUN: {len(refs)} reference + {len(tests)} test captures from the forward "
              f"model. These numbers describe film.py, not a phone.\n")
    else:
        if not args.manifest:
            print("--manifest is required unless --dry-run", file=sys.stderr)
            return 1
        man = pd.read_csv(args.manifest)
        missing = {"capture_path", "role"} - set(man.columns)
        if missing:
            print(f"manifest is missing column(s) {sorted(missing)}; see "
                  f"data/real_recapture/manifest_template.csv", file=sys.stderr)
            return 1
        refs, tests, unreadable = [], [], []
        for row in man.itertuples():
            img = _load(str(row.capture_path))
            if img is None:
                unreadable.append(str(row.capture_path))
                continue
            if str(row.role).lower().startswith("ref"):
                refs.append(img)
            else:
                tests.append((img, {k: getattr(row, k) for k in man.columns if k != "Index"}))
        if unreadable:
            print(f"{len(unreadable)} capture(s) could not be read, e.g. {unreadable[:3]}")
        if not refs:
            print("no reference captures in the manifest (role=reference). The print's densities "
                  "cannot be measured without them, and nothing downstream means anything.",
                  file=sys.stderr)
            return 1
        print(f"{len(refs)} reference + {len(tests)} test captures\n")

    # ------------------------------------------------------ stage 1: characterise
    truth = RC.characterize(refs, ph)
    if truth["n_captures_used"] == 0:
        print("no reference capture could be rectified. Every one of them failed to show a "
              "complete collimation border -- reshoot with the whole sheet in frame.",
              file=sys.stderr)
        for r in truth["rejected"]:
            print(f"  rejected: {r}", file=sys.stderr)
        return 1
    spread = truth["reference_reproducibility_od"]
    print(f"characterised from {truth['n_captures_used']} reference capture(s); "
          f"they agree to {spread:.4f} OD (median across regions)")
    if truth["rejected"]:
        print(f"  {len(truth['rejected'])} reference capture(s) rejected: "
              f"{[r.get('reason', r) for r in truth['rejected']][:3]}")

    # ------------------------------------------------------------ stage 2: score
    rows, disc_rows = [], []
    for i, (photo, meta) in enumerate(tests):
        res = RC.score_capture(photo, ph, truth, spec=spec, meta=meta)
        if not res.get("ok"):
            print(f"  [{i}] not scored: {res.get('reason', 'rectification failed')}")
            rows.append({**meta, "ok": False, "reason": res.get("reason", "rectification failed")})
            continue
        conf = RC.confusion(res["detectability"], rose_k=args.rose_k)
        flat = {
            **meta, "ok": True, "coverage": res["coverage"],
            **{f"tone_{k}": v for k, v in res["tone"].items()},
            **{f"scale_{k}": v for k, v in res["scale"].items()},
            **{f"blur_{k}": v for k, v in res["blur"].items()},
            **{f"veil_{k}": v for k, v in res["veil"].items()},
            **{f"cert_{k}": v for k, v in res["certificate"].items()},
            **{f"disc_{k}": v for k, v in conf.items()},
        }
        rows.append(flat)
        tag = {k: meta.get(k) for k in ("capture_path", "phone_model", "condition")}
        disc_rows.extend({**tag, **d} for d in res["detectability"])
        print(f"  [{i}] {meta.get('condition', '')}: coverage {res['coverage']}, "
              f"tone bias {res['tone'].get('bias_od', float('nan')):+.3f} OD "
              f"(after offset {res['tone'].get('rmse_od_after_offset', float('nan')):.3f}), "
              f"blur {res['blur']['estimator_sigma_px_canonical']:.2f} vs edge "
              f"{res['blur']['phantom_edge_sigma_px_canonical']:.2f} px, "
              f"discs agree {conf['agreement_rate']:.2f}, unsafe {conf['unsafe_rate']:.2f}")

    df = pd.DataFrame(rows)
    discs = pd.DataFrame(disc_rows)
    df.to_csv(out / "captures.csv", index=False)
    discs.to_csv(out / "discs.csv", index=False)
    (out / "characterization.json").write_text(json.dumps(truth, indent=2))

    scored = df[df["ok"]] if "ok" in df.columns else df
    if scored.empty:
        print("\nno test capture could be scored.", file=sys.stderr)
        return 1

    # ---------------------------------------------------------------- the gates
    all_discs = [d for d in disc_rows if "dprime" in d]
    overall = RC.confusion(all_discs, rose_k=args.rose_k)
    tone_rmse = float(np.nanmedian(scored.get("tone_rmse_od_after_offset", pd.Series([np.nan]))))
    ratio = (scored["blur_estimator_sigma_px_canonical"] / scored["blur_phantom_edge_sigma_px_canonical"]
             if "blur_estimator_sigma_px_canonical" in scored else pd.Series([np.nan]))
    blur_ratio = float(np.nanmedian(ratio))
    lo, hi = GATES["blur_ratio_band"]
    checks = {
        "reference_spread": (spread <= GATES["max_reference_spread_od"],
                             f"{spread:.4f} OD <= {GATES['max_reference_spread_od']}"),
        "tone_after_offset": (tone_rmse <= GATES["tone_rmse_after_offset_od"],
                              f"{tone_rmse:.4f} OD <= {GATES['tone_rmse_after_offset_od']}"),
        "blur_agreement": (lo <= blur_ratio <= hi, f"{blur_ratio:.2f} in [{lo}, {hi}]"),
        "clear_violations": (overall["clear_violation_rate"] <= GATES["max_clear_violation_rate"],
                             f"{overall['clear_violation_rate']:.3f} <= "
                             f"{GATES['max_clear_violation_rate']}"),
    }

    print("\n" + "=" * 70)
    print(f"captures scored                  : {len(scored)} of {len(tests)}")
    print(f"discs scored                     : {overall['n_discs']}")
    print(f"certificate vs matched filter    : {overall['agreement_rate']:.2f} agreement")
    print(f"  cleared but invisible          : {overall['predicted_detectable_but_invisible']} "
          f"({overall['unsafe_rate']:.3f}) -- reported, not gated")
    print(f"  of those, clear violations     : {overall['clear_violations']} "
          f"({overall['clear_violation_rate']:.3f})"
          + (f"  e.g. {overall['clear_violation_examples']}" if overall["clear_violations"] else ""))
    print(f"  called insufficient but visible: {overall['predicted_insufficient_but_visible']} "
          f"({overall['conservative_rate']:.3f})")
    print("-" * 70)
    ok = True
    for name, (passed, detail) in checks.items():
        ok &= bool(passed)
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<22} {detail}")
    print("=" * 70)

    summary = {"gates": {k: {"passed": bool(v[0]), "detail": v[1]} for k, v in checks.items()},
               "overall_discs": overall, "reference_spread_od": spread,
               "tone_rmse_after_offset_od": tone_rmse, "blur_ratio": blur_ratio,
               "n_captures_scored": len(scored), "dry_run": bool(args.dry_run)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}/captures.csv, discs.csv, characterization.json, summary.json")
    if args.dry_run:
        print("\nDRY RUN -- film.py on both sides. The gates above say the analysis works, "
              "not that the estimator does.")
        return 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
