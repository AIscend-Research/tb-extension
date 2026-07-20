#!/usr/bin/env python3
"""Scan the data directory and write data/manifest.csv.

Expected layout (only the cohorts you actually have need to be present):

    data/montgomery/CXR_png/MCUCXR_####_X.png
    data/shenzhen/CXR_png/CHNCXR_####_X.png
    data/niaid/labels.csv            columns: image_path,label   (paths under data/niaid/)
    data/rsna/stage_2_train_labels.csv + data/rsna/stage_2_train_images/*.dcm

See docs/DATA.md for where each cohort comes from and how to arrange it.

    python scripts/build_manifest.py --data-root data
    python scripts/build_manifest.py --synthetic       # fake manifest, no images needed
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xctb.data.manifest import (
    build_manifest,
    from_nlm_filenames,
    from_label_csv,
    synthetic_manifest,
    class_balance_table,
)


def collect_sources(data_root: str):
    sources = []
    mont = os.path.join(data_root, "montgomery", "CXR_png")
    if os.path.isdir(mont):
        sources.append(from_nlm_filenames(mont, "montgomery"))
        print(f"  found montgomery under {mont}")

    shen = os.path.join(data_root, "shenzhen", "CXR_png")
    if os.path.isdir(shen):
        sources.append(from_nlm_filenames(shen, "shenzhen"))
        print(f"  found shenzhen under {shen}")

    niaid_csv = os.path.join(data_root, "niaid", "labels.csv")
    if os.path.isfile(niaid_csv):
        sources.append(
            from_label_csv(
                niaid_csv,
                image_root=os.path.join(data_root, "niaid"),
                cohort="niaid",
                path_col="image_path",
                label_col="label",
                positive_value=1,
            )
        )
        print(f"  found niaid from {niaid_csv}")

    rsna_csv = os.path.join(data_root, "rsna", "stage_2_train_labels.csv")
    if os.path.isfile(rsna_csv):
        sources.append(
            from_label_csv(
                rsna_csv,
                image_root=os.path.join(data_root, "rsna", "stage_2_train_images"),
                cohort="rsna",
                path_col="patientId",
                label_col="Target",
                positive_value=1,
                path_suffix=".dcm",
            )
        )
        print(f"  found rsna from {rsna_csv}")
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--synthetic", action="store_true", help="write a fake manifest for testing")
    args = ap.parse_args()

    if args.synthetic:
        print("Building a SYNTHETIC manifest (no real images).")
        manifest = synthetic_manifest()
    else:
        print(f"Scanning {args.data_root} ...")
        sources = collect_sources(args.data_root)
        if not sources:
            sys.exit(
                "No cohorts found. Check --data-root and the layout in docs/DATA.md, "
                "or use --synthetic to test the pipeline without images."
            )
        manifest = build_manifest(sources)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    manifest.to_csv(args.out, index=False)
    print(f"\nWrote {len(manifest)} rows to {args.out}\n")

    table = class_balance_table(manifest)
    print(table.to_string(index=False))
    if table["single_class"].any():
        bad = table.loc[table["single_class"], "cohort"].tolist()
        print(
            f"\nWARNING: single-class cohort(s): {bad}. "
            "Leave-one-cohort-out on these is degenerate (no both-class test, and "
            "cohort predicts label). Read the confounding note in docs/DATA.md before "
            "including them as held-out folds."
        )


if __name__ == "__main__":
    main()
