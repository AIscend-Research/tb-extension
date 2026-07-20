import numpy as np

from xctb.eval.metrics import auroc, sensitivity_specificity, accuracy
from xctb.eval.deferral import (
    risk_coverage_curve,
    accuracy_at_coverage,
    generalization_gap_recovery,
    coverage_to_recover,
)
from xctb.calibration import fit_temperature, apply_temperature, expected_calibration_error


def _oracle_case(n=400, err=0.2, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    wrong = rng.random(n) < err
    prob = np.where(
        wrong,
        np.where(y == 1, rng.uniform(0, 0.5, n), rng.uniform(0.5, 1, n)),
        np.where(y == 1, rng.uniform(0.5, 1, n), rng.uniform(0, 0.5, n)),
    )
    unc = wrong.astype(float) + rng.uniform(0, 0.01, n)
    return y, prob, unc, wrong


def test_metrics_basic():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    assert auroc(y, score) == 1.0
    sens, spec = sensitivity_specificity(y, score, 0.5)
    assert sens == 1.0 and spec == 1.0
    assert accuracy(y, score, 0.5) == 1.0


def test_auroc_single_class_is_nan():
    y = np.array([1, 1, 1])
    score = np.array([0.2, 0.7, 0.9])
    assert np.isnan(auroc(y, score))


def test_full_coverage_accuracy_matches_error_rate():
    y, prob, unc, wrong = _oracle_case(err=0.2)
    acc = accuracy_at_coverage(y, prob, unc, 1.0)
    assert abs((1 - acc) - wrong.mean()) < 1e-9


def test_oracle_recovers_gap():
    y, prob, unc, _ = _oracle_case(err=0.2)
    rec = generalization_gap_recovery(0.99, y, prob, unc, coverages=(1.0, 0.8))
    row = [r for r in rec["rows"] if r["coverage"] == 0.8][0]
    assert row["accuracy"] > 0.98
    assert row["recovery_rate"] > 0.99


def test_random_uncertainty_recovers_less():
    y, prob, unc, _ = _oracle_case(err=0.2, seed=3)
    rng = np.random.default_rng(7)
    rand = rng.random(len(y))
    oracle = generalization_gap_recovery(0.99, y, prob, unc, coverages=(1.0, 0.8))["rows"][-1]
    noise = generalization_gap_recovery(0.99, y, prob, rand, coverages=(1.0, 0.8))["rows"][-1]
    assert noise["recovery_rate"] < oracle["recovery_rate"]


def test_no_gap_gives_nan_recovery():
    y, prob, unc, _ = _oracle_case(err=0.2)
    acc_full = accuracy_at_coverage(y, prob, unc, 1.0)
    rec = generalization_gap_recovery(acc_full, y, prob, unc)  # acc_id == cross-cohort
    assert np.isnan(rec["rows"][0]["recovery_rate"])
    assert coverage_to_recover(acc_full, y, prob, unc, 0.9) is None


def test_aurc_in_unit_interval():
    y, prob, unc, _ = _oracle_case()
    _, risk, aurc = risk_coverage_curve(y, prob, unc)
    assert 0.0 <= aurc <= 1.0
    assert np.all((risk >= 0) & (risk <= 1))


def test_temperature_scaling_reduces_ece():
    rng = np.random.default_rng(0)
    m = 800
    lab = (rng.random(m) < 0.5).astype(int)
    base = 0.8 * (2 * lab - 1) + rng.normal(0, 1.0, m)  # moderate-accuracy score
    pos = 4.0 * base                                    # inflated -> overconfident
    logits = np.stack([np.zeros(m), pos], axis=1)
    ece_before = expected_calibration_error(1 / (1 + np.exp(-pos)), lab)
    T = fit_temperature(logits, lab)
    ece_after = expected_calibration_error(apply_temperature(logits, T), lab)
    assert ece_after <= ece_before + 1e-6
    assert T > 1.0  # overconfident logits need softening
