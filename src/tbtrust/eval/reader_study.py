"""The second-reader claim, turned into a study that can measure it.

`docs/phase1_framing.md` motivates the uncertainty head as learning *"would a
second reader be called on this case?"*, and `docs/LIMITATIONS.md` §4 concedes
the obvious: that is an analogy, not a measurement. No clinician has labelled
anything in this project, so "the model defers on the films a radiologist would
also flag" is asserted rather than shown.

This module is the instrument that closes it. It does three separable things,
and only the third needs a radiologist:

1. **Sampling.** `stratified_sample` draws the films to send out, balanced over
   the joint of (physics margin, learned uncertainty) rather than drawn at
   random. Random sampling spends most of a reader's budget on the easy bulk of
   the corpus, where both signals agree and the answer is uninformative; the
   study's whole question lives in the *discordant* cells, where the certificate
   says the photograph cannot carry the finding but the classifier is confident,
   or the reverse. Balanced allocation buys those cells at the cost of making
   the sample unrepresentative, so every row carries a `sampling_weight` and
   every estimator here is weighted -- the corpus-level quantity is recoverable,
   the naive unweighted one would be a design artifact.

2. **Analysis, pre-registered.** `analyze` takes the returned ratings and
   computes the quantities `docs/READER_STUDY.md` commits to in advance:
   weighted AUC of each signal against "would you seek a second opinion",
   agreement between the deployed deferral policy and the readers' referral set
   (kappa, and Gwet's AC1 because referral is rare and kappa collapses under
   skewed marginals), and the two-question split -- *refer* (the finding is
   ambiguous) versus *retake* (the photograph is inadequate) -- which is the
   clinical form of the same split `physics/triage.py` already makes. The
   pre-registration matters more than usual here: with four signals, two
   questions and nine strata there are enough contrasts to find something.

3. **The ceiling, measurable today.** Radiologists do not agree with each other
   about second reads; reported inter-reader kappa on chest-film abnormality is
   routinely in the 0.4-0.6 range. That caps what *any* model can score against
   a single reader's ratings, and the cap is not a caveat to mention in the
   discussion -- it is a number this module computes before the study runs.
   `reader_noise_ceiling` gives an oracle that knows the latent referral
   propensity exactly the same noisy labels the model will face and reports the
   AUC it achieves. `design_power` then answers the question that decides
   whether the study is worth running: at this n and this many readers, what
   effect is detectable, and how much of the gap to the ceiling is reader noise
   rather than model failure.

The falsification is stated up front and lives in the doc: if the certificate
margin's weighted AUC against reader referral does not clear 0.5 by more than
its CI, the physics track is measuring something readers do not care about, and
the second-reader framing should be dropped from the paper rather than softened.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# The instrument. Reproduced verbatim in docs/READER_STUDY.md; the constants
# are here so the analysis and the form cannot drift apart.
# --------------------------------------------------------------------------

REFER_QUESTION = (
    "Reading this film as presented, would you seek a second opinion before "
    "reporting it?"
)
REFER_SCALE = (
    "1 = no, confident single read",
    "2 = probably not",
    "3 = borderline",
    "4 = probably yes",
    "5 = yes, I would not report this alone",
)
#: >= this on the 1-5 scale counts as "would refer" for the binary contrasts.
REFER_BINARY_CUT = 4

ADEQUACY_QUESTION = (
    "Is the image quality adequate to report, or would you ask for the film to "
    "be re-photographed?"
)
ADEQUACY_SCALE = ("adequate", "borderline", "inadequate -- retake")

#: Free text, one of these plus an optional note. Mirrors `physics/triage.py`'s
#: reasons so the two vocabularies can be cross-tabulated.
REASON_CODES = (
    "blur", "glare", "shadow", "contrast", "cropping", "artefact",
    "ambiguous_finding", "subtle_finding", "other",
)

#: Signals scored against the ratings. Keys are columns in the rows table.
SIGNALS = {
    "physics_score": "certificate margin with abstentions ranked worst",
    "margin_db": "physics certificate margin (lower = less information)",
    "mc_std": "MC-dropout predictive spread",
    "learned_confidence": "max(p, 1-p) on the calibrated probability",
    "mc_confidence": "MC-dropout confidence",
}
#: True when *larger* values of the signal should mean *more* referral.
SIGNAL_HIGHER_MEANS_REFER = {
    "physics_score": True,
    "margin_db": False,
    "mc_std": True,
    "learned_confidence": False,
    "mc_confidence": False,
}


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

@dataclass
class SampleDesign:
    """A drawn reader-study sample plus everything needed to un-draw it."""

    rows: list[dict]                     # one per photograph shown
    n_strata: tuple[int, int]
    population: dict                     # cell -> corpus count
    allocation: dict                     # cell -> drawn count
    n_repeats: int
    seed: int
    notes: dict = field(default_factory=dict)

    @property
    def n_cases(self) -> int:
        return sum(1 for r in self.rows if not r["is_repeat"])

    @property
    def n_shown(self) -> int:
        return len(self.rows)


def _quantile_stratum(values: np.ndarray, k: int) -> np.ndarray:
    """Index 0..k-1 by within-corpus quantile; -1 for values that do not exist.

    The -1 is not a tidiness detail. On the real corpus 141 of 600 photographs
    carry no margin at all -- `limiting_factor == "no_fiducials"`, the
    certificate declining to answer because it could not find the L/R marker to
    calibrate against. An earlier version let `searchsorted` place those NaNs in
    the top stratum, i.e. filed every abstention as "most information survived",
    which is the exact opposite of what an abstention means. They get their own
    stratum instead, and the study asks readers about them too: whether the
    certificate's abstentions land on films a reader also calls inadequate is a
    result, not a nuisance.
    """
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    out = np.full(len(v), -1, dtype=int)
    if ok.any():
        edges = np.quantile(v[ok], np.linspace(0, 1, k + 1)[1:-1])
        out[ok] = np.searchsorted(edges, v[ok], side="left").astype(int)
    return out


def physics_referral_score(margin_db, abstained=None) -> np.ndarray:
    """One score for the certificate, abstentions included, oriented up = refer.

    Scoring the raw margin drops abstentions on the floor: they are NaN, every
    estimator here skips non-finite scores, and the physics track would be
    graded only on the films where it was willing to speak. That flatters it.
    The deployed policy defers an abstention, so the study scores it that way --
    abstentions are ranked below the worst finite margin, which is the ordering
    `physics/triage.py` actually acts on.
    """
    m = np.asarray(margin_db, dtype=float)
    ab = (~np.isfinite(m)) if abstained is None else (
        np.asarray(abstained).astype(bool) | ~np.isfinite(m))
    finite = m[np.isfinite(m)]
    worst = float(finite.min()) if finite.size else 0.0
    score = -np.where(ab, worst - 1.0, m)     # negate: low margin = high referral
    return score


def _case_id(key: str, severity: float, salt: str) -> str:
    h = hashlib.sha256(f"{salt}|{key}|{severity:.4f}".encode()).hexdigest()
    return f"C{int(h[:8], 16) % 10**6:06d}"


def _balanced_assignment(buckets, film_cells, per_cell, rng, film_col):
    """Assign at most one cell per film, as evenly across cells as is possible.

    Greedy allocation is not good enough here and the failure is not subtle: with
    film uniqueness enforced across cells, taking cells in any fixed order lets
    early cells spend films that a later, scarcer cell was the only one able to
    use. On the real corpus that produced cells of 10 and cells of 2 -- a
    "balanced" design that is nothing of the kind, underpowered precisely in the
    discordant cells the study exists to buy.

    So it is solved as what it is: a degree-constrained bipartite matching
    between films and cells, by augmenting paths (Kuhn's algorithm), filled one
    level at a time so every cell reaches k before any cell reaches k+1. When a
    cell has no free film left, the augmenting search asks whether some other
    cell holding one of its candidates could swap to a different film, and takes
    it if so.
    """
    assigned: dict = {c: [] for c in buckets}
    owner: dict = {}                      # film -> cell currently holding it

    def _augment(cell, seen: set) -> bool:
        films = [f for f, cs in film_cells.items() if cell in cs and f not in seen]
        rng.shuffle(films)
        for f in films:
            seen.add(f)
            held_by = owner.get(f)
            if held_by is None:
                owner[f] = cell
                assigned[cell].append(f)
                return True
            if _augment(held_by, seen):   # that cell can free this film
                assigned[held_by].remove(f)
                owner[f] = cell
                assigned[cell].append(f)
                return True
        return False

    for level in range(1, per_cell + 1):
        cells = list(buckets)
        rng.shuffle(cells)
        for cell in cells:
            if len(assigned[cell]) < level:
                _augment(cell, set())

    drawn: list[dict] = []
    for cell, films in assigned.items():
        by_film: dict = {}
        for r in buckets[cell]:
            by_film.setdefault(r[film_col], []).append(r)
        for f in films:
            cand = by_film[f]
            drawn.append(dict(cand[int(rng.integers(len(cand)))]))
    return {c: len(f) for c, f in assigned.items()}, drawn


def stratified_sample(
    rows,
    n_cases: int = 120,
    n_margin_strata: int = 3,
    n_uncertainty_strata: int = 3,
    uncertainty_col: str = "mc_std",
    margin_col: str = "margin_db",
    film_col: str = "key",
    repeat_fraction: float = 0.1,
    seed: int = 0,
    salt: str = "tbtrust-reader-study",
) -> SampleDesign:
    """Draw a balanced sample over the (margin x uncertainty) grid.

    `rows` is any sequence of dicts with the columns above -- in practice the
    output of `scripts/physics_deferral_real.py`, one row per (film, severity),
    which is one photograph and therefore one thing to show a reader.

    Balanced allocation, not proportional: every cell gets the same number of
    films regardless of how rare it is in the corpus. The rare cells are the
    discordant ones and they carry the entire contrast, so proportional
    allocation would spend ~85% of the reader's time confirming that easy films
    are easy. `sampling_weight = population(cell) / drawn(cell)` restores the
    corpus estimand; do not drop it.

    A `repeat_fraction` of the drawn films is shown twice, in shuffled positions,
    so intra-reader repeatability is measurable from the study itself rather than
    assumed. Without it a low model-vs-reader agreement has two explanations and
    no way to separate them.
    """
    rows = [dict(r) for r in rows]
    if not rows:
        raise ValueError("no rows to sample from")

    marg = np.asarray([float(r[margin_col]) for r in rows])
    unc = np.asarray([float(r[uncertainty_col]) for r in rows])
    s_m = _quantile_stratum(marg, n_margin_strata)
    s_u = _quantile_stratum(unc, n_uncertainty_strata)

    population: dict = {}
    buckets: dict = {}
    for i, r in enumerate(rows):
        cell = (int(s_m[i]), int(s_u[i]))
        r["stratum_margin"], r["stratum_uncertainty"] = cell
        r["cell"] = f"m{'A' if cell[0] < 0 else cell[0]}u{cell[1]}"
        population[r["cell"]] = population.get(r["cell"], 0) + 1
        buckets.setdefault(r["cell"], []).append(r)

    rng = np.random.default_rng(seed)
    # Cells counted after assignment, not from the grid: an abstain stratum
    # appears only if the corpus contains abstentions, and budgeting for a grid
    # that does not exist overspends the reader by a whole row of cells.
    per_cell = max(1, n_cases // max(1, len(buckets)))

    # Which cells each film could serve. A film has five severities and they land
    # in different cells, so a film is a *choice* of cell, not a fixed member of
    # one -- and it may be spent on only one, because showing one reader the same
    # film twice at two severities leaks the answer and inflates agreement.
    film_cells: dict = {}
    for cell, pool in buckets.items():
        for r in pool:
            film_cells.setdefault(r[film_col], set()).add(cell)

    allocation, drawn = _balanced_assignment(buckets, film_cells, per_cell, rng,
                                             film_col)
    shortfall = {c: per_cell - n for c, n in allocation.items() if n < per_cell}

    for r in drawn:
        r["sampling_weight"] = population[r["cell"]] / max(1, allocation[r["cell"]])
        r["case_id"] = _case_id(str(r[film_col]), float(r.get("severity", 0.0)), salt)
        r["is_repeat"] = False

    n_rep = round(repeat_fraction * len(drawn))
    idx = rng.choice(len(drawn), size=min(n_rep, len(drawn)), replace=False)
    repeats = []
    for i in idx:
        dup = dict(drawn[int(i)])
        dup["is_repeat"] = True
        dup["repeat_of"] = dup["case_id"]
        dup["case_id"] = dup["case_id"] + "R"
        dup["sampling_weight"] = 0.0        # repeats inform reliability, not prevalence
        repeats.append(dup)

    shown = drawn + repeats
    order = rng.permutation(len(shown))
    for pos, i in enumerate(order):
        shown[int(i)]["display_order"] = pos
    shown.sort(key=lambda r: r["display_order"])

    return SampleDesign(
        rows=shown,
        n_strata=(n_margin_strata, n_uncertainty_strata),
        population=population,
        allocation=allocation,
        n_repeats=len(repeats),
        seed=seed,
        notes={
            "uncertainty_col": uncertainty_col,
            "margin_col": margin_col,
            "balanced": True,
            "per_cell_target": per_cell,
            # Non-empty means some cell ran out of unclaimed films. The weights
            # still hold, but the design is no longer balanced and the study is
            # underpowered exactly where it can least afford to be.
            "shortfall": shortfall,
        },
    )


def rating_sheet(design: SampleDesign) -> tuple[list[dict], list[dict]]:
    """Split the sample into (blinded sheet the reader fills, unblinding key).

    The sheet carries no label, no clinic, no severity and no model output --
    a reader who can see that the model deferred is no longer an independent
    measurement of anything.
    """
    sheet, key = [], []
    for r in design.rows:
        sheet.append({
            "display_order": r["display_order"],
            "case_id": r["case_id"],
            "refer_1_to_5": "",
            "adequacy": "",
            "reason_code": "",
            "note": "",
        })
        key.append({k: v for k, v in r.items() if k != "display_order"} | {
            "display_order": r["display_order"]})
    return sheet, key


# --------------------------------------------------------------------------
# Weighted estimators
# --------------------------------------------------------------------------

def weighted_auc(scores, positive, weights=None) -> float:
    """Mann-Whitney AUC with sampling weights and ties counted as half.

    Weighted because the sample is balanced by design and the corpus is not.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(positive).astype(bool)
    w = np.ones_like(s) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(s) & (w > 0)
    s, y, w = s[ok], y[ok], w[ok]
    if y.all() or not y.any():
        return float("nan")
    if np.all(w == w[0]):
        # Equal weights: the rank identity gives the same number in O(n log n)
        # instead of O(n^2). The power simulation calls this millions of times.
        from scipy.stats import rankdata
        r = rankdata(s)
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    sp, wp = s[y], w[y]
    sn, wn = s[~y], w[~y]
    gt = (sp[:, None] > sn[None, :]).astype(float)
    eq = (sp[:, None] == sn[None, :]).astype(float)
    num = float((wp[:, None] * wn[None, :] * (gt + 0.5 * eq)).sum())
    den = float(wp.sum() * wn.sum())
    return num / den if den > 0 else float("nan")


def cluster_bootstrap_ci(
    scores, positive, weights=None, clusters=None, n_boot: int = 2000,
    alpha: float = 0.05, seed: int = 0,
) -> tuple[float, float, float]:
    """(point, lo, hi) for the weighted AUC, resampling *films*, not photographs.

    The same film appears in the corpus at several severities and, across
    readers, several times; resampling rows would treat those as independent and
    hand back an interval that is too tight to be honest.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(positive).astype(bool)
    w = np.ones_like(s) if weights is None else np.asarray(weights, dtype=float)
    cl = np.arange(len(s)) if clusters is None else np.asarray(clusters)
    point = weighted_auc(s, y, w)
    uniq, inv = np.unique(cl, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(len(uniq), size=len(uniq))
        idx = np.concatenate([members[int(g)] for g in pick])
        a = weighted_auc(s[idx], y[idx], w[idx])
        if np.isfinite(a):
            draws.append(a)
    if len(draws) < 20:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Analytic AUC standard error -- used for the power table, not the report.

    The study reports the cluster bootstrap. This closed form exists because
    power has to be computed for values of n that have not been collected.
    """
    if n_pos < 1 or n_neg < 1 or not np.isfinite(auc):
        return float("nan")
    a = float(np.clip(auc, 1e-6, 1 - 1e-6))
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a))
    return float(np.sqrt(max(var, 0.0) / (n_pos * n_neg)))


def cohen_kappa(a, b, weights=None) -> float:
    """Two binary ratings, chance-corrected the usual way."""
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    w = np.ones(len(a)) if weights is None else np.asarray(weights, dtype=float)
    tot = w.sum()
    if tot <= 0:
        return float("nan")
    po = float(w[a == b].sum() / tot)
    pa, pb = float(w[a].sum() / tot), float(w[b].sum() / tot)
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def gwet_ac1(a, b, weights=None) -> float:
    """Gwet's AC1: the same agreement, chance-corrected without kappa's paradox.

    Referral is rare. With a skewed marginal, kappa's expected agreement runs
    close to the observed agreement and kappa falls towards zero even when the
    two raters agree on 90% of films -- the well-known first paradox. Reporting
    both is the only honest way to state a rare-event agreement, and if they
    disagree the AC1 is the one to believe.
    """
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    w = np.ones(len(a)) if weights is None else np.asarray(weights, dtype=float)
    tot = w.sum()
    if tot <= 0:
        return float("nan")
    po = float(w[a == b].sum() / tot)
    pi = 0.5 * (float(w[a].sum() / tot) + float(w[b].sum() / tot))
    pe = 2 * pi * (1 - pi)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def icc21(ratings: np.ndarray) -> float:
    """ICC(2,1), two-way random effects, absolute agreement, single rater.

    `ratings` is (n_cases, n_readers). This is the reliability of *one* reader's
    ordinal score, which is exactly the quantity that attenuates every
    correlation the study reports.
    """
    x = np.asarray(ratings, dtype=float)
    n, k = x.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = x.mean()
    row, col = x.mean(axis=1), x.mean(axis=0)
    msr = k * ((row - grand) ** 2).sum() / (n - 1)
    msc = n * ((col - grand) ** 2).sum() / (k - 1)
    resid = x - row[:, None] - col[None, :] + grand
    mse = (resid ** 2).sum() / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den > 0 else float("nan")


def spearman_brown(icc_single: float, k: int) -> float:
    """Reliability of a k-reader consensus given one reader's reliability."""
    if not np.isfinite(icc_single) or icc_single <= 0:
        return float("nan")
    return float(k * icc_single / (1 + (k - 1) * icc_single))


# --------------------------------------------------------------------------
# The ceiling and the power, both computable before any radiologist is booked
# --------------------------------------------------------------------------

@dataclass
class ReaderModel:
    """A generative stand-in for a panel of readers.

    Not a claim about radiologists. It is the null-and-alternative machinery the
    power calculation needs, with every knob set from a literature range rather
    than from this project's data: `icc_single` is single-reader reliability on
    chest-film calls (0.4-0.6 is the routinely reported band), `refer_rate` is
    the prevalence of "I want a second read", and `signal_r` is the correlation
    between the model signal and the latent referral propensity -- the effect
    size being powered for, i.e. the thing the study is supposed to measure.
    """

    icc_single: float = 0.5
    refer_rate: float = 0.25
    signal_r: float = 0.45
    n_readers: int = 3
    reader_bias_sd: float = 0.25


def simulate_ratings(signal, model: ReaderModel, seed: int = 0):
    """Ratings a panel matching `model` would return, given a model signal.

    Latent propensity z = signal_r * standardised(signal) + independent case
    variation; each reader sees z plus a personal bias and per-film noise sized
    to hit `icc_single`. Referral is the top `refer_rate` of each reader's own
    latent scale, so readers differ in threshold as well as in noise -- which is
    what actually drives disagreement between real readers.

    Returns (binary refer matrix, ordinal 1-5 matrix, latent propensity), each
    (n_cases, n_readers) except the last.
    """
    s = np.asarray(signal, dtype=float)
    rng = np.random.default_rng(seed)
    z = (s - np.nanmean(s)) / (np.nanstd(s) + 1e-12)
    r = float(np.clip(model.signal_r, 0.0, 0.99))
    latent = r * z + np.sqrt(max(1 - r * r, 1e-9)) * rng.standard_normal(len(s))

    icc = float(np.clip(model.icc_single, 1e-3, 0.999))
    # var(latent) = 1 by construction; reader noise sized so ICC(2,1) ~ icc.
    sigma_e = np.sqrt((1 - icc) / icc)
    bias = model.reader_bias_sd * rng.standard_normal(model.n_readers)
    obs = latent[:, None] + bias[None, :] + sigma_e * rng.standard_normal(
        (len(s), model.n_readers))

    refer = np.zeros_like(obs, dtype=bool)
    ordinal = np.zeros_like(obs)
    for j in range(model.n_readers):
        cut = np.quantile(obs[:, j], 1 - model.refer_rate)
        refer[:, j] = obs[:, j] > cut
        qs = np.quantile(obs[:, j], [0.2, 0.4, 0.6, 0.8])
        ordinal[:, j] = 1 + np.searchsorted(qs, obs[:, j], side="left")
    return refer, ordinal, latent


def reader_noise_ceiling(
    n_cases: int = 120, model: ReaderModel | None = None, n_sim: int = 400,
    seed: int = 0,
) -> dict:
    """The best AUC an *oracle* model could score against these readers.

    The oracle is handed the latent referral propensity itself -- a model that
    has solved the problem completely. It still does not reach 1.0, because the
    labels it is scored against are one reader's noisy call. That number is the
    ceiling, and the honest denominator for whatever the real signals score:
    a measured AUC of 0.72 against a ceiling of 0.78 is a near-solved problem
    reported as a mediocre one if the ceiling is left out.

    Reported for a single reader and for the majority vote of the panel, since
    consensus is cheaper reliability than a better model.
    """
    model = model or ReaderModel()
    rng = np.random.default_rng(seed)
    single, panel = [], []
    for _ in range(n_sim):
        sig = rng.standard_normal(n_cases)
        refer, _ord, latent = simulate_ratings(
            sig, model, seed=int(rng.integers(1 << 31)))
        single.append(weighted_auc(latent, refer[:, 0]))
        vote = refer.mean(axis=1) > 0.5
        if vote.any() and not vote.all():
            panel.append(weighted_auc(latent, vote))
    return {
        "n_cases": n_cases,
        "icc_single": model.icc_single,
        "n_readers": model.n_readers,
        "refer_rate": model.refer_rate,
        "auc_ceiling_single_reader": float(np.mean(single)),
        "auc_ceiling_majority_vote": float(np.mean(panel)) if panel else float("nan"),
        "consensus_reliability": spearman_brown(model.icc_single, model.n_readers),
    }


def design_power(
    n_cases: int = 120, model: ReaderModel | None = None, n_sim: int = 400,
    alpha: float = 0.05, seed: int = 0, n_boot: int = 200,
) -> dict:
    """Empirical power of the pre-registered test, at this n and this effect.

    The test is the one the doc commits to: reject "the signal is unrelated to
    referral" when the cluster-bootstrap lower bound on the weighted AUC clears
    0.5. Power is the fraction of simulated studies that reject when the effect
    is real; the same routine run at `signal_r = 0` returns the realised type-I
    rate, which is reported next to it because a bootstrap interval on a
    weighted AUC is not guaranteed to be calibrated and should be checked rather
    than trusted.
    """
    model = model or ReaderModel()
    rng = np.random.default_rng(seed)

    def _run(r: float) -> float:
        m = ReaderModel(icc_single=model.icc_single, refer_rate=model.refer_rate,
                        signal_r=r, n_readers=model.n_readers,
                        reader_bias_sd=model.reader_bias_sd)
        hits = 0
        for _ in range(n_sim):
            sig = rng.standard_normal(n_cases)
            refer, _o, _l = simulate_ratings(sig, m, seed=int(rng.integers(1 << 31)))
            vote = refer.mean(axis=1) > 0.5
            if vote.all() or not vote.any():
                continue
            _p, lo, _hi = cluster_bootstrap_ci(
                sig, vote, n_boot=n_boot, alpha=alpha, seed=int(rng.integers(1 << 31)))
            hits += int(np.isfinite(lo) and lo > 0.5)
        return hits / n_sim

    return {
        "n_cases": n_cases,
        "signal_r": model.signal_r,
        "icc_single": model.icc_single,
        "n_readers": model.n_readers,
        "power": _run(model.signal_r),
        "type_i_at_null": _run(0.0),
    }


def minimum_detectable_auc(
    n_cases: int, refer_rate: float = 0.25, alpha: float = 0.05, power: float = 0.8,
) -> float:
    """Smallest AUC separable from 0.5 at this n -- the analytic sanity check.

    Solved by bisection on the Hanley-McNeil SE, which assumes independent
    photographs. The study's films are clustered, so treat this as a lower bound
    on the n required and the simulation above as the operative answer.
    """
    from scipy.stats import norm

    n_pos = max(1, round(n_cases * refer_rate))
    n_neg = max(1, n_cases - n_pos)
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    lo, hi = 0.5, 0.999
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        se = hanley_mcneil_se(mid, n_pos, n_neg)
        if (mid - 0.5) / se >= z:
            hi = mid
        else:
            lo = mid
    return float(hi)


# --------------------------------------------------------------------------
# The analysis the study runs when the ratings come back
# --------------------------------------------------------------------------

def analyze(
    rows,
    refer_binary,
    ordinal=None,
    signals=None,
    weights=None,
    clusters=None,
    model_defers=None,
    adequacy_inadequate=None,
    physics_says_retake=None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """The pre-registered read-out, given ratings for the drawn sample.

    `refer_binary` is (n_cases,) consensus or (n_cases, n_readers); a matrix is
    reduced by majority vote and also yields the inter-reader block. Everything
    is weighted by `sampling_weight` and bootstrapped over `clusters` (films).

    Three blocks come back:

    * `signals` -- weighted AUC + CI for each model signal against referral. The
      headline. `margin_db` is the one the physics track lives or dies on.
    * `policy` -- the deployed defer/answer decision against the readers'
      referral set: agreement, kappa, AC1, and the two error directions kept
      separate, because deferring a film no reader would refer costs clinic time
      while answering one every reader would refer is the failure that matters.
    * `triage` -- the retake/refer split. `physics/triage.py` calls a film either
      retakeable or referable; this asks whether readers make the same cut on the
      same photographs, which is a sharper test than the referral AUC because it
      is about *which action*, not just *whether something is wrong*.
    """
    rows = [dict(r) for r in rows]
    refer = np.asarray(refer_binary)
    inter = {}
    if refer.ndim == 2:
        if ordinal is not None:
            o = np.asarray(ordinal, dtype=float)
            inter["icc_single_reader"] = icc21(o)
            inter["icc_consensus"] = spearman_brown(icc21(o), o.shape[1])
        pairs = [(i, j) for i in range(refer.shape[1]) for j in range(i + 1, refer.shape[1])]
        inter["pairwise_kappa"] = [cohen_kappa(refer[:, i], refer[:, j]) for i, j in pairs]
        inter["pairwise_ac1"] = [gwet_ac1(refer[:, i], refer[:, j]) for i, j in pairs]
        inter["refer_rate_per_reader"] = [float(refer[:, j].mean())
                                          for j in range(refer.shape[1])]
        y = refer.mean(axis=1) > 0.5
    else:
        y = refer.astype(bool)

    w = (np.asarray([float(r.get("sampling_weight", 1.0)) for r in rows])
         if weights is None else np.asarray(weights, dtype=float))
    cl = (np.asarray([r.get("key", i) for i, r in enumerate(rows)])
          if clusters is None else np.asarray(clusters))
    signals = signals or list(SIGNALS)

    out: dict = {
        "n_cases": len(rows),
        "refer_rate_weighted": float((w * y).sum() / w.sum()) if w.sum() else float("nan"),
        "refer_rate_raw": float(y.mean()),
        "inter_reader": inter,
        "signals": {},
    }

    for name in signals:
        if name not in rows[0]:
            continue
        s = np.asarray([float(r[name]) for r in rows])
        if not SIGNAL_HIGHER_MEANS_REFER.get(name, True):
            s = -s
        point, lo, hi = cluster_bootstrap_ci(
            s, y, weights=w, clusters=cl, n_boot=n_boot, seed=seed)
        out["signals"][name] = {
            "auc": point, "ci_lo": lo, "ci_hi": hi,
            "beats_chance": bool(np.isfinite(lo) and lo > 0.5),
            "oriented": "higher = more referral (sign applied)",
        }

    if model_defers is not None:
        d = np.asarray(model_defers).astype(bool)
        tot = w.sum()
        out["policy"] = {
            "agreement_weighted": float(w[d == y].sum() / tot) if tot else float("nan"),
            "kappa": cohen_kappa(d, y, w),
            "gwet_ac1": gwet_ac1(d, y, w),
            "defer_rate_weighted": float((w * d).sum() / tot) if tot else float("nan"),
            "caught": float((w * (d & y)).sum() / max((w * y).sum(), 1e-12)),
            "answered_but_reader_would_refer":
                float((w * (~d & y)).sum() / max((w * y).sum(), 1e-12)),
            "deferred_but_no_reader_would":
                float((w * (d & ~y)).sum() / max((w * ~y).sum(), 1e-12)),
        }

    if adequacy_inadequate is not None and physics_says_retake is not None:
        a = np.asarray(adequacy_inadequate).astype(bool)
        p = np.asarray(physics_says_retake).astype(bool)
        out["triage"] = {
            "reader_inadequate_rate": float((w * a).sum() / w.sum()) if w.sum() else float("nan"),
            "physics_retake_rate": float((w * p).sum() / w.sum()) if w.sum() else float("nan"),
            "kappa": cohen_kappa(p, a, w),
            "gwet_ac1": gwet_ac1(p, a, w),
            # The clinically wrong action: physics sends a film back for a retake
            # that the reader considered perfectly reportable.
            "retake_requested_on_adequate_film":
                float((w * (p & ~a)).sum() / max((w * ~a).sum(), 1e-12)),
        }
    return out
