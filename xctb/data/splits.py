"""Split construction.

Two ways to split, and the contrast between them is the experiment:

  random_split          shuffle everything, ignore cohort. This is how most
                        papers report accuracy, and it is the optimistic number.

  leave_one_cohort_out  train on N-1 cohorts, test on the cohort the model has
                        never seen. This is the deployment number.

The gap between those two is what the whole project is trying to measure and
then close with calibrated deferral. So the split code is not plumbing, it is
the thing being studied. It also has to refuse to leak: if a held-out cohort's
images sneak into training, the deployment number is silently inflated and the
result is worthless. check_split enforces that.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def leave_one_cohort_out(
    manifest: pd.DataFrame,
    val_fraction: float = 0.15,
    seed: int = 0,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]]:
    """Yield (train, val, test, held_out_cohort) for each cohort in turn.

    test  = every image from the held-out cohort
    val   = a slice carved out of the remaining cohorts, used for model
            selection and for fitting the deferral threshold. It is drawn from
            the *seen* cohorts on purpose: you never get to tune on the held-out
            machine, because in deployment you would not have it.
    train = the rest of the seen cohorts
    """
    cohorts = sorted(manifest["cohort"].unique())
    if len(cohorts) < 2:
        raise ValueError(
            f"leave-one-cohort-out needs at least 2 cohorts, found {cohorts}."
        )
    rng = np.random.default_rng(seed)

    for held_out in cohorts:
        test = manifest[manifest["cohort"] == held_out].reset_index(drop=True)
        seen = manifest[manifest["cohort"] != held_out].reset_index(drop=True)

        # Stratify the val carve-out by (cohort, label) so small cohorts and the
        # minority class do not vanish from validation.
        val_idx = []
        for _, group in seen.groupby(["cohort", "label"]):
            idx = group.index.to_numpy().copy()
            rng.shuffle(idx)
            k = int(round(len(idx) * val_fraction))
            val_idx.extend(idx[:k].tolist())
        val_mask = seen.index.isin(val_idx)
        val = seen[val_mask].reset_index(drop=True)
        train = seen[~val_mask].reset_index(drop=True)

        check_split(train, val, test, held_out)
        yield train, val, test, held_out


def random_split(
    manifest: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cohort-blind stratified split. Produces the in-distribution reference."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for _, group in manifest.groupby(["cohort", "label"]):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        test_idx.extend(idx[:n_test].tolist())
        val_idx.extend(idx[n_test : n_test + n_val].tolist())
        train_idx.extend(idx[n_test + n_val :].tolist())
    take = lambda ix: manifest.loc[ix].reset_index(drop=True)
    return take(train_idx), take(val_idx), take(test_idx)


def check_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    held_out: str,
) -> None:
    """Fail loud on leakage; warn on a degenerate held-out cohort.

    Leakage (held-out cohort appearing in train or val) is a hard error because
    it invalidates the result. A single-class held-out cohort is only a warning,
    because it is a real property of the data (NIAID is almost all TB, RSNA
    almost all normal) that you may still want to report, just carefully.
    """
    import warnings

    train_cohorts = set(train["cohort"].unique())
    val_cohorts = set(val["cohort"].unique())
    if held_out in train_cohorts:
        raise AssertionError(f"Leakage: held-out cohort {held_out!r} is in the training set.")
    if held_out in val_cohorts:
        raise AssertionError(f"Leakage: held-out cohort {held_out!r} is in the validation set.")

    train_paths = set(train["image_path"])
    if train_paths & set(test["image_path"]):
        raise AssertionError("Leakage: some test images also appear in training.")
    if train_paths & set(val["image_path"]):
        raise AssertionError("Leakage: some val images also appear in training.")

    test_classes = set(test["label"].unique())
    if len(test_classes) < 2:
        warnings.warn(
            f"Held-out cohort {held_out!r} has only class {test_classes}. "
            "You can report accuracy on it, but sensitivity/specificity are "
            "undefined for the missing class and the fold is easy to over-read.",
            stacklevel=2,
        )
    if len(set(train["label"].unique())) < 2:
        warnings.warn(
            f"Training set for held-out {held_out!r} has a single class.",
            stacklevel=2,
        )
