"""Quantify the domain shift itself.

Before claiming the cohorts come from different machines, measure it. For a
sample of images per cohort this records brightness (mean pixel), contrast
(pixel std), and native resolution. The resulting table is a concrete "how
different are these sources" number to put in the paper, and a quick way to
notice, for example, that one cohort is 4000x4000 12-bit and another is
1024x1024 8-bit.

Torch-free: PIL + numpy only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def image_stats(path: str) -> dict:
    from PIL import Image

    if str(path).lower().endswith(".dcm"):
        import pydicom

        arr = pydicom.dcmread(path).pixel_array.astype(np.float32)
    else:
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    h, w = arr.shape[:2]
    a = arr / 255.0 if arr.max() > 1.5 else arr
    return {
        "height": int(h),
        "width": int(w),
        "megapixels": round(h * w / 1e6, 3),
        "brightness": float(a.mean()),
        "contrast": float(a.std()),
    }


def cohort_shift_table(manifest: pd.DataFrame, sample_per_cohort: int = 100, seed: int = 0) -> pd.DataFrame:
    """Sample images per cohort and summarise their statistics.

    Returns one row per cohort with median resolution and mean brightness /
    contrast across the sample.
    """
    rng = np.random.default_rng(seed)
    records = []
    for cohort, group in manifest.groupby("cohort"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        picks = idx[: min(sample_per_cohort, len(idx))]
        stats = []
        for i in picks:
            try:
                stats.append(image_stats(manifest.loc[i, "image_path"]))
            except Exception:
                continue  # skip unreadable files rather than crash the whole run
        if not stats:
            continue
        s = pd.DataFrame(stats)
        records.append(
            {
                "cohort": cohort,
                "n_sampled": len(s),
                "median_megapixels": round(float(s["megapixels"].median()), 3),
                "mean_brightness": round(float(s["brightness"].mean()), 4),
                "mean_contrast": round(float(s["contrast"].mean()), 4),
            }
        )
    return pd.DataFrame(records).sort_values("cohort").reset_index(drop=True)
