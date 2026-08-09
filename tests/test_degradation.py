import numpy as np
import pytest
from PIL import Image

from xctb.data.degradation import (
    DEGRADATION_KINDS,
    DEGRADATION_STRATEGIES,
    apply_degradation,
    compose_degradation,
    severity_to_target_uncertainty,
    build_degradation_manifest,
)
from xctb.data.manifest import synthetic_manifest
from xctb.eval.degradation_uncertainty import spearman_correlation, uncertainty_vs_severity


def _checkerboard(size=64):
    arr = np.indices((size, size)).sum(axis=0) % 2 * 255
    return Image.fromarray(arr.astype(np.uint8), mode="L")


@pytest.mark.parametrize("kind", DEGRADATION_KINDS)
def test_zero_severity_is_identity(kind):
    img = _checkerboard()
    out = apply_degradation(img, kind, 0.0, rng=np.random.default_rng(0))
    assert np.array_equal(np.asarray(img), np.asarray(out))


@pytest.mark.parametrize("kind", DEGRADATION_KINDS)
def test_nonzero_severity_changes_image_and_keeps_shape(kind):
    img = _checkerboard()
    out = apply_degradation(img, kind, 0.8, rng=np.random.default_rng(0))
    assert out.size == img.size
    assert out.mode == "L"
    assert not np.array_equal(np.asarray(img), np.asarray(out))


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        apply_degradation(_checkerboard(), "not-a-kind", 0.5)


def test_defocus_blur_reduces_high_frequency_energy():
    img = _checkerboard()
    out = apply_degradation(img, "defocus_blur", 1.0, rng=np.random.default_rng(0))
    arr_in = np.asarray(img, dtype=np.float32)
    arr_out = np.asarray(out, dtype=np.float32)
    # A checkerboard has huge pixel-to-pixel variance; blur must shrink it.
    assert arr_out.std() < arr_in.std()


def test_compose_zero_severity_is_identity():
    img = _checkerboard()
    out, applied = compose_degradation(img, 0.0, rng=np.random.default_rng(0), strategy="full")
    assert np.array_equal(np.asarray(img), np.asarray(out))
    assert set(applied) == set(DEGRADATION_STRATEGIES["full"])
    assert all(v == 0.0 for v in applied.values())


def test_compose_full_applies_every_kind_and_changes_image():
    img = _checkerboard()
    out, applied = compose_degradation(img, 1.0, rng=np.random.default_rng(1), strategy="full")
    assert set(applied) == set(DEGRADATION_KINDS)
    assert all(0.0 <= v <= 1.0 for v in applied.values())
    assert not np.array_equal(np.asarray(img), np.asarray(out))


def test_compose_simple_uses_only_simple_kinds():
    img = _checkerboard()
    _, applied = compose_degradation(img, 1.0, rng=np.random.default_rng(1), strategy="simple")
    assert set(applied) == set(DEGRADATION_STRATEGIES["simple"])


def test_compose_is_reproducible_given_same_seed():
    img = _checkerboard()
    out1, applied1 = compose_degradation(img, 0.6, rng=np.random.default_rng(42), strategy="full")
    out2, applied2 = compose_degradation(img, 0.6, rng=np.random.default_rng(42), strategy="full")
    assert np.array_equal(np.asarray(out1), np.asarray(out2))
    assert applied1 == applied2


def test_unimplemented_strategy_raises_not_implemented():
    for name, kinds in DEGRADATION_STRATEGIES.items():
        if kinds is None:
            with pytest.raises(NotImplementedError):
                compose_degradation(_checkerboard(), 0.5, strategy=name)


def test_unknown_strategy_raises_value_error():
    with pytest.raises(ValueError):
        compose_degradation(_checkerboard(), 0.5, strategy="not-a-strategy")


def test_severity_to_target_uncertainty_is_monotonic_and_clipped():
    assert severity_to_target_uncertainty(0.0) == 0.0
    assert severity_to_target_uncertainty(1.0) == 1.0
    assert severity_to_target_uncertainty(-1.0) == 0.0
    assert severity_to_target_uncertainty(2.0) == 1.0
    assert severity_to_target_uncertainty(0.3) < severity_to_target_uncertainty(0.7)


def test_build_degradation_manifest_tags_every_row():
    m = synthetic_manifest(sizes={"montgomery": 10, "shenzhen": 10}, pos_rates={"montgomery": 0.5, "shenzhen": 0.5})
    severities = (0.0, 0.5, 1.0)
    out = build_degradation_manifest(m, severities=severities, strategy="full", seed=0)

    assert len(out) == len(m) * len(severities)
    for col in ("image_path", "label", "cohort", "degradation_strategy", "degradation_severity", "degradation_seed"):
        assert col in out.columns
    # original clinic/TB-status tags survive the expansion untouched
    assert set(out["cohort"].unique()) == set(m["cohort"].unique())
    assert set(out.loc[out["degradation_severity"] == 0.0, "degradation_strategy"]) == {"none"}
    assert set(out.loc[out["degradation_severity"] > 0.0, "degradation_strategy"]) == {"full"}


def test_spearman_perfect_monotonic():
    a = np.arange(20)
    b = np.arange(20) * 2.0 + 1.0
    assert spearman_correlation(a, b) == pytest.approx(1.0)
    assert spearman_correlation(a, -b) == pytest.approx(-1.0)


def test_spearman_constant_input_is_nan():
    a = np.arange(10)
    b = np.ones(10)
    assert np.isnan(spearman_correlation(a, b))


def test_uncertainty_vs_severity_flags_honest_and_dishonest_models():
    rng = np.random.default_rng(0)
    severity = rng.uniform(0, 1, 300)
    honest_uncertainty = severity_to_target_uncertainty(severity) + rng.normal(0, 0.05, 300)
    dishonest_uncertainty = rng.uniform(0, 1, 300)

    honest = uncertainty_vs_severity(severity, honest_uncertainty)
    dishonest = uncertainty_vs_severity(severity, dishonest_uncertainty)

    assert honest["n"] == 300
    assert honest["spearman_rho"] > 0.8
    assert abs(dishonest["spearman_rho"]) < 0.2
