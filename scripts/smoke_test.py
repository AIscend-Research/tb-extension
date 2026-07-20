#!/usr/bin/env python3
"""End-to-end sanity check of the parts that make this project novel, with no
GPU, no images, and no torch. Run it first to confirm the environment and the
core logic work before you touch real data or training.

    python scripts/smoke_test.py

It builds a synthetic manifest, constructs leave-one-cohort-out splits and checks
they do not leak, fabricates predictions whose uncertainty is correlated with
error, and then verifies the deferral and calibration math behaves the way the
paper assumes: deferring uncertain cases recovers accuracy, and temperature
scaling reduces calibration error.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from xctb.data.manifest import synthetic_manifest, class_balance_table, validate_manifest
from xctb.data.splits import leave_one_cohort_out, random_split
from xctb.eval.metrics import binary_report
from xctb.eval.deferral import (
    risk_coverage_curve,
    accuracy_at_coverage,
    generalization_gap_recovery,
    coverage_to_recover,
)
from xctb.calibration import fit_temperature, apply_temperature, expected_calibration_error


def section(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    rng = np.random.default_rng(0)

    section("1. manifest")
    manifest = synthetic_manifest()
    assert not validate_manifest(manifest), "synthetic manifest should validate"
    print(class_balance_table(manifest).to_string(index=False))

    section("2. leave-one-cohort-out splits (leakage checks)")
    n_folds = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # single-class cohorts warn on purpose
        for train, val, test, held in leave_one_cohort_out(manifest, seed=0):
            assert held not in set(train["cohort"]), "held-out cohort leaked into train"
            assert held not in set(val["cohort"]), "held-out cohort leaked into val"
            assert not (set(train["image_path"]) & set(test["image_path"])), "image leakage"
            assert set(test["cohort"]) == {held}, "test must be exactly the held-out cohort"
            print(f"  hold out {held:<11} train={len(train):>4} val={len(val):>4} test={len(test):>4}  ok")
            n_folds += 1
    assert n_folds == manifest["cohort"].nunique(), "one fold per cohort expected"

    section("3. random split (in-distribution reference)")
    tr, va, te = random_split(manifest, seed=0)
    assert len(tr) + len(va) + len(te) == len(manifest), "random split must partition the manifest"
    print(f"  train={len(tr)} val={len(va)} test={len(te)}")

    section("4. deferral: oracle uncertainty recovers the gap")
    # Fabricate a held-out cohort where 20% of predictions are wrong, and make
    # uncertainty perfectly rank those errors first. Deferring them should lift
    # the kept accuracy to ~1.0, i.e. recover the whole gap.
    n = 500
    y_true = (rng.random(n) < 0.5).astype(int)
    wrong = rng.random(n) < 0.20
    # prob is on the correct side unless "wrong"; uncertainty tracks wrongness (+ noise)
    prob = np.where(
        wrong,
        np.where(y_true == 1, rng.uniform(0.0, 0.5, n), rng.uniform(0.5, 1.0, n)),
        np.where(y_true == 1, rng.uniform(0.5, 1.0, n), rng.uniform(0.0, 0.5, n)),
    )
    uncertainty = wrong.astype(float) + rng.uniform(0, 0.05, n)

    acc_full = accuracy_at_coverage(y_true, prob, uncertainty, 1.0)
    coverage, risk, aurc = risk_coverage_curve(y_true, prob, uncertainty)
    print(f"  full-coverage accuracy = {acc_full:.3f}  (expected ~0.80)")
    print(f"  AURC = {aurc:.4f}  (low is good)")
    assert risk[0] <= risk[-1] + 1e-9, "risk at low coverage should not exceed risk at full coverage"

    acc_id = 0.99
    rec = generalization_gap_recovery(acc_id, y_true, prob, uncertainty)
    print(f"  gap = {rec['gap']:.3f}")
    # at coverage 0.80 we keep exactly the correct cases -> accuracy ~1.0
    row80 = next(r for r in rec["rows"] if abs(r["coverage"] - 0.8) < 1e-6)
    print(f"  coverage 0.80 -> accuracy {row80['accuracy']:.3f}, "
          f"recovery_rate {row80['recovery_rate']:.3f}")
    assert row80["accuracy"] > 0.98, "deferring the (oracle) errors should give ~1.0 accuracy"
    assert row80["recovery_rate"] > 0.99, "oracle deferral should recover ~all of the gap"

    reach = coverage_to_recover(acc_id, y_true, prob, uncertainty, target_fraction=0.9)
    assert reach is not None, "should be able to recover 90% of the gap with oracle uncertainty"
    print(f"  recover 90% of the gap by deferring {reach[1]*100:.0f}% of cases")

    section("5. deferral: random uncertainty recovers little")
    rand_unc = rng.random(n)
    rec_rand = generalization_gap_recovery(acc_id, y_true, prob, rand_unc)
    row80_rand = next(r for r in rec_rand["rows"] if abs(r["coverage"] - 0.8) < 1e-6)
    print(f"  coverage 0.80 -> recovery_rate {row80_rand['recovery_rate']:.3f} (should be near 0)")
    assert row80_rand["recovery_rate"] < row80["recovery_rate"], \
        "useless uncertainty should recover less than oracle uncertainty"

    section("6. calibration: temperature scaling reduces ECE")
    # Build a moderately accurate classifier (a noisy score, ~78% accuracy),
    # then inflate its logits by a known factor so it is overconfident: the
    # predictions do not change, but the probabilities are pushed toward 0/1.
    # Temperature scaling should recover roughly that inflation factor (T > 1)
    # and bring ECE down.
    m = 800
    lab = (rng.random(m) < 0.5).astype(int)
    base = 0.8 * (2 * lab - 1) + rng.normal(0, 1.0, m)   # signal + noise
    inflate = 4.0
    pos_logit = inflate * base                            # overconfident logits
    logits = np.stack([np.zeros(m), pos_logit], axis=1)

    ece_before = expected_calibration_error(1 / (1 + np.exp(-pos_logit)), lab)
    T = fit_temperature(logits, lab)
    ece_after = expected_calibration_error(apply_temperature(logits, T), lab)
    print(f"  base accuracy ~ {accuracy_at_coverage(lab, 1/(1+np.exp(-pos_logit)), np.zeros(m), 1.0):.3f}")
    print(f"  fitted T = {T:.3f} (expect > 1, near the inflation factor {inflate})")
    print(f"  ECE {ece_before:.4f} -> {ece_after:.4f}")
    assert T > 1.0, "overconfident logits should call for softening (T > 1)"
    assert ece_after <= ece_before + 1e-6, "temperature scaling should not worsen ECE"

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
