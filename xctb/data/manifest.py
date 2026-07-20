"""Build one unified manifest across all cohorts.

A manifest is a pandas DataFrame with exactly three required columns:

    image_path : str   absolute or repo-relative path to a single CXR image
    label      : int   0 = normal, 1 = TB / abnormal
    cohort     : str   which source the image came from (montgomery, shenzhen, ...)

The cohort column is the whole point. We train with it (domain-invariance and
the cohort-conditional layers read it) and we split on it (leave-one-cohort-out
holds one value out entirely). Get the tagging right here and the rest of the
pipeline stays honest about which machine each image came from.

Adding a new cohort means writing one small loader that returns a DataFrame with
those three columns, then registering it in build_manifest.
"""

from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd

# Canonical cohort names. Keep these stable; splits, configs and logged results
# all key off these strings.
COHORTS = ["montgomery", "shenzhen", "niaid", "rsna"]

REQUIRED_COLUMNS = ["image_path", "label", "cohort"]


# --------------------------------------------------------------------------- #
# Per-cohort loaders
# --------------------------------------------------------------------------- #
def from_nlm_filenames(image_dir: str, cohort: str, pattern: str = "*.png") -> pd.DataFrame:
    """Montgomery and Shenzhen encode the label in the filename.

    Both use the template NAME_####_X.png where X is 0 (normal) or 1 (abnormal).
    Montgomery files start MCUCXR_, Shenzhen files start CHNCXR_.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(image_dir, pattern))):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            label = int(stem.split("_")[-1])
        except (ValueError, IndexError):
            # Skip anything that does not match the convention rather than guess.
            continue
        if label not in (0, 1):
            continue
        rows.append({"image_path": path, "label": label, "cohort": cohort})
    if not rows:
        raise FileNotFoundError(
            f"No labelled images found under {image_dir!r} matching {pattern!r}. "
            "Check the path and that filenames follow NAME_####_X.png."
        )
    return pd.DataFrame(rows)


def from_label_csv(
    csv_path: str,
    image_root: str,
    cohort: str,
    path_col: str,
    label_col: str,
    positive_value=1,
    path_suffix: str = "",
) -> pd.DataFrame:
    """Generic loader for cohorts that ship a labels CSV (NIAID export, RSNA).

    Example, RSNA normals only:
        from_label_csv("stage_2_train_labels.csv", "stage_2_train_images",
                       cohort="rsna", path_col="patientId", label_col="Target",
                       positive_value=1, path_suffix=".dcm")
        # then keep label == 0 rows if you only want RSNA normals (see DATA.md)
    """
    df = pd.read_csv(csv_path)
    for col in (path_col, label_col):
        if col not in df.columns:
            raise KeyError(f"Column {col!r} not in {csv_path} (has {list(df.columns)}).")
    out = pd.DataFrame(
        {
            "image_path": [
                os.path.join(image_root, str(p) + path_suffix) for p in df[path_col]
            ],
            "label": (df[label_col] == positive_value).astype(int).to_numpy(),
            "cohort": cohort,
        }
    )
    # A patient can appear on several rows (RSNA has one row per bounding box).
    return out.drop_duplicates(subset="image_path").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_manifest(sources: list[pd.DataFrame], validate: bool = True) -> pd.DataFrame:
    """Concatenate per-cohort DataFrames into the unified manifest."""
    if not sources:
        raise ValueError("build_manifest received no sources.")
    manifest = pd.concat(sources, ignore_index=True)
    manifest = manifest[REQUIRED_COLUMNS].copy()
    manifest["label"] = manifest["label"].astype(int)
    manifest["cohort"] = manifest["cohort"].astype(str)
    if validate:
        problems = validate_manifest(manifest)
        if problems:
            raise ValueError("Manifest failed validation:\n  - " + "\n  - ".join(problems))
    return manifest.reset_index(drop=True)


def synthetic_manifest(
    sizes: dict[str, int] | None = None,
    pos_rates: dict[str, float] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """A fake manifest with real column shape, for tests and the smoke run.

    Paths are placeholders that point at nothing. Defaults mimic the real
    quirk that cohorts differ a lot in size and positive rate.
    """
    sizes = sizes or {"montgomery": 138, "shenzhen": 662, "niaid": 300, "rsna": 800}
    pos_rates = pos_rates or {"montgomery": 0.42, "shenzhen": 0.51, "niaid": 0.95, "rsna": 0.05}
    rng = np.random.default_rng(seed)
    rows = []
    for cohort, n in sizes.items():
        p = pos_rates.get(cohort, 0.5)
        labels = (rng.random(n) < p).astype(int)
        for i, y in enumerate(labels):
            rows.append(
                {
                    "image_path": f"/synthetic/{cohort}/img_{i:05d}.png",
                    "label": int(y),
                    "cohort": cohort,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sanity checks and reporting
# --------------------------------------------------------------------------- #
def validate_manifest(manifest: pd.DataFrame) -> list[str]:
    """Return a list of problems. Empty list means the manifest is well formed."""
    problems: list[str] = []
    for col in REQUIRED_COLUMNS:
        if col not in manifest.columns:
            problems.append(f"missing column {col!r}")
    if problems:
        return problems  # nothing else is safe to check yet

    if manifest["image_path"].isna().any():
        problems.append("some image_path values are missing")
    if manifest["image_path"].duplicated().any():
        n = int(manifest["image_path"].duplicated().sum())
        problems.append(f"{n} duplicate image_path values")
    bad_labels = set(manifest["label"].unique()) - {0, 1}
    if bad_labels:
        problems.append(f"labels outside 0/1: {sorted(bad_labels)}")
    if manifest["cohort"].isna().any():
        problems.append("some cohort values are missing")
    return problems


def class_balance_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Per-cohort counts and positive rate, sorted by cohort.

    The 'single_class' column is the one to watch. A cohort with only positives
    or only negatives cannot be scored for both sensitivity and specificity when
    it is the held-out set, and it makes the cohort label a near-perfect proxy
    for the class label during training. See DATA.md for why RSNA and NIAID
    tend to trip this.
    """
    g = manifest.groupby("cohort")["label"]
    table = pd.DataFrame(
        {
            "n": g.size(),
            "pos": g.sum(),
        }
    )
    table["neg"] = table["n"] - table["pos"]
    table["pos_rate"] = (table["pos"] / table["n"]).round(3)
    table["single_class"] = (table["pos"] == 0) | (table["neg"] == 0)
    return table.reset_index().sort_values("cohort").reset_index(drop=True)
