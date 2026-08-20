"""The reader-study rig: the places a wrong answer would look like a right one.

`scripts/reader_study.py` draws a sample no radiologist has rated yet, so nothing
downstream can catch an error by looking wrong. These pin the parts that would
quietly bias the study instead: the balanced sample's weights failing to recover
the corpus, an abstention being filed as "most information survived", and kappa
being trusted on a rare event.
"""

from __future__ import annotations

import numpy as np
import pytest

from tbtrust.eval import reader_study as RS


def _corpus(n=600, seed=0):
    """A corpus with a real signal and a realistic slug of abstentions."""
    rng = np.random.default_rng(seed)
    margin = rng.normal(-11.0, 5.0, n)
    margin[rng.random(n) < 0.235] = np.nan          # no fiducials detected
    return [{"key": f"f{i // 5}", "severity": (i % 5) * 0.25,
             "margin_db": float(margin[i]), "mc_std": float(rng.gamma(2, 0.02)),
             "abstained": bool(not np.isfinite(margin[i]))}
            for i in range(n)]


# --- the sampler ----------------------------------------------------------

def test_abstentions_get_their_own_stratum_not_the_top_one():
    """The bug this was written for: NaN margins searchsorted into stratum k-1.

    An abstention means the certificate could not be computed. Filing it with
    the *best* margins inverts its meaning, and nothing downstream would notice
    -- the sample would just quietly over-represent abstentions as good films.
    """
    s = RS._quantile_stratum(np.array([1.0, 2.0, 3.0, np.nan, np.nan]), 3)
    assert set(s[3:]) == {-1}
    assert (s[:3] >= 0).all()


def test_strata_are_populated_and_weights_recover_the_corpus():
    corpus = _corpus()
    d = RS.stratified_sample(corpus, n_cases=120, seed=1)
    assert len(d.allocation) == 12                  # 3x3 grid + an abstain row
    # Balanced by design: no cell more than one film off any other.
    counts = sorted(d.allocation.values())
    assert counts[-1] - counts[0] <= 1
    # ... and the weights put the corpus back together.
    drawn = [r for r in d.rows if not r["is_repeat"]]
    assert sum(r["sampling_weight"] for r in drawn) == pytest.approx(len(corpus), rel=0.02)


def test_one_photograph_per_film_and_repeats_carry_no_weight():
    d = RS.stratified_sample(_corpus(), n_cases=120, repeat_fraction=0.1, seed=2)
    drawn = [r for r in d.rows if not r["is_repeat"]]
    assert len({r["key"] for r in drawn}) == len(drawn)   # no film shown twice
    reps = [r for r in d.rows if r["is_repeat"]]
    assert reps and all(r["sampling_weight"] == 0.0 for r in reps)
    assert all(r["repeat_of"] in {x["case_id"] for x in drawn} for r in reps)


def test_the_sheet_leaks_nothing():
    d = RS.stratified_sample(_corpus(), n_cases=60, seed=3)
    sheet, key = RS.rating_sheet(d)
    leaks = {"margin_db", "mc_std", "label", "clinic", "severity", "abstained"}
    assert not (set(sheet[0]) & leaks)
    assert set(key[0]) & leaks                       # the key still has them


def test_abstentions_are_scored_as_worst_not_dropped():
    """Scoring the raw margin would grade the physics only where it spoke."""
    m = np.array([-5.0, -1.0, np.nan])
    s = RS.physics_referral_score(m, abstained=[False, False, True])
    assert np.isfinite(s).all()
    assert s[2] > s[0] > s[1]                        # abstain ranks most-refer


# --- the estimators -------------------------------------------------------

def test_weighted_auc_fast_path_matches_the_pairwise_definition():
    rng = np.random.default_rng(4)
    s, y = rng.standard_normal(200), rng.random(200) < 0.3
    w = np.full(200, 1.0) + 1e-9 * np.arange(200)    # forces the O(n^2) branch
    assert RS.weighted_auc(s, y) == pytest.approx(RS.weighted_auc(s, y, w), abs=1e-6)


def test_weighted_auc_undoes_a_biased_sample():
    """The reason every estimator here is weighted.

    Oversample the positives 10x and the unweighted AUC drifts; the weighted one
    has to come back to the corpus value.
    """
    rng = np.random.default_rng(5)
    n = 4000
    y = rng.random(n) < 0.2
    s = rng.standard_normal(n) + 1.2 * y
    truth = RS.weighted_auc(s, y)
    keep = np.flatnonzero(y | (rng.random(n) < 0.1))
    w = np.where(y[keep], 1.0, 10.0)
    assert RS.weighted_auc(s[keep], y[keep], w) == pytest.approx(truth, abs=0.03)


def test_auc_ties_count_as_half():
    assert RS.weighted_auc(np.zeros(6), [1, 1, 1, 0, 0, 0]) == pytest.approx(0.5)


def test_bootstrap_clusters_by_film_and_widens_the_interval():
    """Resampling rows instead of films buys a dishonestly tight interval."""
    rng = np.random.default_rng(6)
    films = np.repeat(np.arange(40), 5)
    per_film = rng.standard_normal(40)
    y = np.repeat(per_film > 0, 5)
    s = per_film[films] + 0.3 * rng.standard_normal(200)
    _p, lo_c, hi_c = RS.cluster_bootstrap_ci(s, y, clusters=films, n_boot=400, seed=0)
    _p, lo_r, hi_r = RS.cluster_bootstrap_ci(s, y, n_boot=400, seed=0)
    assert (hi_c - lo_c) > (hi_r - lo_r)


def test_gwet_ac1_survives_the_rare_event_where_kappa_collapses():
    """Kappa's first paradox, which is why both are reported."""
    a = np.zeros(100, dtype=bool)
    a[:5] = True
    b = a.copy()
    b[4] = False
    b[5] = True                                      # 98% agreement, rare event
    assert (a == b).mean() == pytest.approx(0.98)
    assert RS.cohen_kappa(a, b) < 0.8                # kappa punishes the skew
    assert RS.gwet_ac1(a, b) > 0.95                  # AC1 does not


def test_icc_and_spearman_brown_move_the_right_way():
    rng = np.random.default_rng(7)
    latent = rng.standard_normal(200)
    tight = latent[:, None] + 0.3 * rng.standard_normal((200, 3))
    loose = latent[:, None] + 2.0 * rng.standard_normal((200, 3))
    assert RS.icc21(tight) > RS.icc21(loose)
    assert RS.spearman_brown(0.5, 3) == pytest.approx(0.75)
    assert RS.spearman_brown(0.5, 1) == pytest.approx(0.5)


# --- the ceiling and the end-to-end read-out ------------------------------

def test_the_ceiling_is_below_one_and_rises_with_reader_agreement():
    """An oracle scored against noisy labels cannot reach 1.0. That is the point."""
    lo = RS.reader_noise_ceiling(n_cases=150, model=RS.ReaderModel(icc_single=0.4),
                                 n_sim=60, seed=0)
    hi = RS.reader_noise_ceiling(n_cases=150, model=RS.ReaderModel(icc_single=0.7),
                                 n_sim=60, seed=0)
    assert 0.5 < lo["auc_ceiling_single_reader"] < 1.0
    assert hi["auc_ceiling_single_reader"] > lo["auc_ceiling_single_reader"]
    # A panel is cheaper reliability than a better model.
    assert hi["auc_ceiling_majority_vote"] > hi["auc_ceiling_single_reader"]


def test_analyze_recovers_a_planted_signal_and_reports_the_ceiling_honestly():
    corpus = _corpus(seed=8)
    d = RS.stratified_sample(corpus, n_cases=120, seed=8)
    rows = [r for r in d.rows if not r["is_repeat"]]
    for r in rows:
        r["physics_score"] = float(RS.physics_referral_score(
            [r["margin_db"]], [r["abstained"]])[0])
    sig = np.array([r["physics_score"] for r in rows])
    refer, ordinal, _l = RS.simulate_ratings(
        sig, RS.ReaderModel(signal_r=0.7, n_readers=3), seed=8)

    res = RS.analyze(rows, refer, ordinal=ordinal, signals=["physics_score", "mc_std"],
                     n_boot=300, seed=0)
    assert res["signals"]["physics_score"]["beats_chance"]
    # mc_std was not what the readers were simulated from; it must not light up.
    assert not res["signals"]["mc_std"]["beats_chance"]
    assert 0.0 < res["inter_reader"]["icc_single_reader"] < 1.0
    assert len(res["inter_reader"]["pairwise_kappa"]) == 3


def test_analyze_weights_by_the_sampling_weight_by_default():
    """Silently dropping the weight column would change the estimand."""
    rows = [{"key": f"f{i}", "s": float(i), "sampling_weight": 1.0 if i < 50 else 9.0}
            for i in range(100)]
    y = np.array([i >= 50 for i in range(100)])
    res = RS.analyze(rows, y, signals=["s"], n_boot=50)
    assert res["refer_rate_weighted"] > res["refer_rate_raw"]
