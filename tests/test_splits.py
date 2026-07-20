import warnings

import pytest

from xctb.data.manifest import synthetic_manifest, validate_manifest, class_balance_table
from xctb.data.splits import leave_one_cohort_out, random_split, check_split


def test_manifest_validates():
    m = synthetic_manifest()
    assert validate_manifest(m) == []


def test_loco_no_leakage_and_one_fold_per_cohort():
    m = synthetic_manifest()
    seen_holdouts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train, val, test, held in leave_one_cohort_out(m, seed=0):
            assert held not in set(train["cohort"])
            assert held not in set(val["cohort"])
            assert set(test["cohort"]) == {held}
            assert not (set(train["image_path"]) & set(test["image_path"]))
            assert not (set(val["image_path"]) & set(test["image_path"]))
            seen_holdouts.append(held)
    assert sorted(seen_holdouts) == sorted(m["cohort"].unique())


def test_random_split_partitions():
    m = synthetic_manifest()
    tr, va, te = random_split(m, seed=1)
    total = len(tr) + len(va) + len(te)
    assert total == len(m)
    paths = set(tr["image_path"]) | set(va["image_path"]) | set(te["image_path"])
    assert len(paths) == len(m)  # no overlap


def test_single_class_holdout_warns():
    # rsna here is all-negative -> held out, it should warn (degenerate fold)
    m = synthetic_manifest(
        sizes={"a": 100, "b": 100, "rsna": 100},
        pos_rates={"a": 0.5, "b": 0.5, "rsna": 0.0},
    )
    assert bool(class_balance_table(m).set_index("cohort").loc["rsna", "single_class"])
    with pytest.warns(UserWarning):
        for _ in leave_one_cohort_out(m, seed=0):
            pass


def test_check_split_detects_leakage():
    m = synthetic_manifest(sizes={"a": 20, "b": 20}, pos_rates={"a": 0.5, "b": 0.5})
    train = m[m["cohort"] == "a"]
    test = m[m["cohort"] == "b"]
    with pytest.raises(AssertionError):
        # held-out "b" present in the training frame -> leakage
        check_split(m, train.iloc[:2], test, held_out="b")
