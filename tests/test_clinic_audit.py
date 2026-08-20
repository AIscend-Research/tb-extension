"""The checks that stand between a new source and a reported fold.

Both audits fail in the same direction if they are wrong: they return a
comfortable number. An overlap check with a threshold inside the known-different
distribution says "half your clinic is duplicated" and gets switched off; one
with a threshold too low says "clean" about a fold that is entirely re-hosted. A
confound check that cannot see a planted shortcut clears a fold that a classifier
will solve without looking at the lungs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tbtrust.data import audit as A
from tbtrust.data import sources as S


def _nlm_images():
    """Real Montgomery/Shenzhen paths, when the raw data happens to be present."""
    import pandas as pd

    m = Path("data/processed/manifest.csv")
    if not m.exists():
        return None
    df = pd.read_csv(m)
    mont = [p for p in df[df["clinic"] == "montgomery"]["path"] if Path(p).exists()]
    shen = [p for p in df[df["clinic"] == "shenzhen"]["path"] if Path(p).exists()]
    return (mont, shen) if len(mont) >= 12 and len(shen) >= 12 else None


_NLM_IMAGES = _nlm_images()


# --- the registry ---------------------------------------------------------

def test_the_kaggle_aggregate_is_never_offered_as_a_fold():
    """The convenient one that would put NLM images on both sides of a split."""
    agg = S.BY_KEY["kaggle_tb_aggregate"]
    assert agg.verdict == "not_a_fold"
    assert not agg.is_candidate_fold
    assert agg not in S.candidate_folds()


def test_every_candidate_fold_actually_has_both_classes():
    for s in S.candidate_folds():
        assert s.both_classes, f"{s.key} offered as a fold without both classes"


def test_single_class_sources_are_excluded():
    for s in S.CANDIDATES:
        if s.both_classes is False:
            assert not s.is_candidate_fold


def test_unverified_fields_are_surfaced_not_buried():
    rows = {r["key"]: r for r in S.summary_rows()}
    assert "unverified_fields" in rows["nitrd_da"]
    # The registry claims nothing was verified for NITRD that was not.
    assert set(rows["nitrd_da"]["unverified_fields"]) <= set(S.BY_KEY["nitrd_da"].evidence)


# --- perceptual hashing ---------------------------------------------------

def _chest_like(seed, size=256):
    """A smooth, low-frequency image with per-seed structure.

    Chest films are far more alike than photographs -- that is the whole reason
    the threshold needed calibrating -- but they are not *identical*, and a
    fixture whose images differ only by a noise seed would make every pair a
    duplicate and prove nothing. Each seed gets its own field geometry, rib
    spacing and exposure, so two seeds are as different as two patients.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size] / size
    cx, cy = rng.uniform(0.35, 0.65, 2)
    spread = rng.uniform(0.06, 0.25)
    base = rng.uniform(70, 150) + rng.uniform(40, 90) * np.exp(
        -((x - cx) ** 2 + (y - cy) ** 2) / spread)
    base += rng.uniform(8, 30) * np.sin(rng.uniform(3, 12) * np.pi * y
                                        + rng.uniform(0, 6.28))
    base += rng.uniform(10, 40) * np.exp(-((x - rng.uniform(0.1, 0.9)) ** 2) / 0.02)
    base += 6 * rng.standard_normal((size, size))
    return np.clip(base, 0, 255).astype(np.uint8)


def test_hash_survives_the_rescale_and_recompression_it_must_catch(tmp_path):
    from PIL import Image

    p = tmp_path / "a.png"
    Image.fromarray(_chest_like(1, 512)).save(p)
    original = A.hash_image_file(str(p))
    rebundled = A.dhash(A.simulate_rebundle(str(p)))
    assert A.hamming(original, rebundled) <= A.DUPLICATE_THRESHOLD


def test_duplicates_and_different_images_separate(tmp_path):
    """The invariant the threshold rests on: every duplicate is closer than
    every non-duplicate.

    Asserted as a separation rather than against `DUPLICATE_THRESHOLD`, because
    the absolute gap is a property of the *source*, not of the hash. These
    synthetic films are smoother and more alike than real radiographs, so they
    separate by a smaller margin than the NLM sets do (which is why the real
    calibration lives in the test below, and in the doc). A test that pinned the
    absolute number here would be pinning the fixture.
    """
    from PIL import Image

    paths = []
    for i in range(6):
        p = tmp_path / f"{i}.png"
        Image.fromarray(_chest_like(100 + i, 512)).save(p)
        paths.append(str(p))
    hs = [A.hash_image_file(p) for p in paths]
    different = [A.hamming(hs[i], hs[j])
                 for i in range(len(hs)) for j in range(i + 1, len(hs))]
    duplicate = [A.hamming(A.hash_image_file(p), A.dhash(A.simulate_rebundle(p)))
                 for p in paths]
    assert max(duplicate) < min(different)


@pytest.mark.skipif(not _NLM_IMAGES, reason="needs data/raw NLM images")
def test_the_shipped_threshold_holds_on_the_real_sources():
    """The calibration the defaults were set from, re-run.

    This is the claim `docs/SOURCES.md` makes -- duplicates within 2 bits, the
    nearest known-different NLM pair 26 bits away -- and the shipped threshold
    has to sit inside that gap or the audit is running on a number nobody
    checked.
    """
    mont, shen = _NLM_IMAGES
    res = A.calibrate_threshold(mont[:12], mont[:12], shen[:12])
    assert res["usable"]
    assert res["positive_max"] < A.DUPLICATE_THRESHOLD < res["negative_min"]


def test_the_default_hash_size_is_the_calibrated_one():
    """8x8 was the first choice and it reported half of Montgomery as duplicated."""
    assert A.HASH_SIZE >= 16
    assert 0 < A.DUPLICATE_THRESHOLD < A.HASH_SIZE * A.HASH_SIZE // 4


def test_calibration_reports_unusable_rather_than_splitting_an_overlap(tmp_path):
    """No threshold works when the distributions touch, and it must say so."""
    from PIL import Image

    same, diff_a, diff_b = [], [], []
    for i in range(4):
        for bucket, seed in ((same, 1), (diff_a, 1), (diff_b, 1)):
            p = tmp_path / f"{id(bucket)}_{i}.png"
            Image.fromarray(_chest_like(seed, 256)).save(p)   # all identical
            bucket.append(str(p))
    res = A.calibrate_threshold(same, diff_a, diff_b, size=8)
    # "Different" here is actually the same image, so the gap collapses.
    assert res["usable"] is False
    assert res["recommended_threshold"] is None


def test_find_overlap_catches_a_planted_rehost(tmp_path):
    from PIL import Image

    a_paths, b_paths = [], []
    for i in range(4):
        pa = tmp_path / f"a{i}.png"
        Image.fromarray(_chest_like(200 + i, 512)).save(pa)
        a_paths.append(str(pa))
    # b is three fresh images plus one re-host of a0.
    for i in range(3):
        pb = tmp_path / f"b{i}.png"
        Image.fromarray(_chest_like(300 + i, 512)).save(pb)
        b_paths.append(str(pb))
    leak = tmp_path / "b_leak.png"
    Image.fromarray(A.simulate_rebundle(a_paths[0])).save(leak)
    b_paths.append(str(leak))

    # Threshold from this fixture's own separation, not the shipped default:
    # these synthetic films are more alike than real radiographs, so the NLM
    # threshold sits inside their known-different distribution.
    cal = A.calibrate_threshold(a_paths, a_paths, b_paths[:3])
    assert cal["usable"], "fixture must separate before the audit can be tested"
    rep = A.find_overlap(a_paths, b_paths, threshold=cal["recommended_threshold"])

    assert rep.n_overlapping == 1
    assert rep.pairs[0][1].endswith("b_leak.png")
    assert rep.verdict() == "overlapping"
    clean = A.find_overlap(a_paths[1:], b_paths[:3],
                           threshold=cal["recommended_threshold"])
    assert clean.verdict() == "clean"


# --- source confound ------------------------------------------------------

def _features_with_shortcut(n, sep):
    """Two groups whose brightness differs by `sep` and nothing else."""
    rng = np.random.default_rng(0)
    out, y = [], []
    for label in (0, 1):
        for _ in range(n):
            f = {k: float(rng.standard_normal()) for k in A.CAPTURE_FEATURES}
            f["mean"] = 100 + sep * label + rng.standard_normal()
            out.append(f)
            y.append(label)
    return out, y


def test_a_planted_capture_shortcut_is_detected():
    feats, y = _features_with_shortcut(60, sep=8.0)
    res = A.source_confound(feats, y, seed=0)
    assert res["auc"] > 0.95
    assert res["top_feature"] == "mean"
    assert A.confound_verdict(res["auc"]) == "confounded"


def test_no_shortcut_reads_as_chance():
    feats, y = _features_with_shortcut(80, sep=0.0)
    res = A.source_confound(feats, y, seed=0)
    assert 0.35 < res["auc"] < 0.65
    assert A.confound_verdict(res["auc"]) == "acceptable"


def test_confound_is_cross_validated_not_in_sample():
    """In sample, nine features on a small fold would condemn every fold."""
    feats, y = _features_with_shortcut(12, sep=0.0)
    res = A.source_confound(feats, y, seed=0)
    assert res["auc"] < 0.9        # an in-sample fit would be far higher


def test_single_class_input_is_unmeasurable_not_chance():
    feats, _ = _features_with_shortcut(20, sep=0.0)
    res = A.source_confound(feats, [1] * len(feats))
    assert not np.isfinite(res["auc"])
    assert A.confound_verdict(res["auc"]) == "unmeasurable"


def test_resize_changes_the_geometry_features_only(tmp_path):
    """The dimensions give-away must be removable, so it can be told apart."""
    from PIL import Image

    p = tmp_path / "x.png"
    Image.fromarray(_chest_like(7, 500)).save(p)
    raw = A.capture_features(str(p))
    small = A.capture_features(str(p), resize=224)
    assert raw["width"] == 500 and small["width"] == 224
    assert small["mean"] == pytest.approx(raw["mean"], rel=0.05)
