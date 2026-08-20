# Radiologist agreement: the study, pre-registered

`docs/phase1_framing.md` motivates the uncertainty head as learning *"would a
second reader be called on this case?"*. `docs/LIMITATIONS.md` §4 concedes that
this is an analogy: no clinician has labelled anything in this project, so
"the model defers on the films a radiologist would also flag" is a claim with no
measurement behind it. This document is the protocol that measures it, and
`src/tbtrust/eval/reader_study.py` + `scripts/reader_study.py` are the rig.

Everything here except the readers themselves is runnable today, and the parts
that are runnable today already returned three results that change what the
study should be — they are in §5, and one of them says a piece of the study
cannot be run at this operating point at all.

## 1. The question, stated so it can fail

> Do the photographs the system flags — high learned uncertainty, or a physics
> certificate margin at or below zero — correspond to films a radiologist would
> actually refer for a second read?

Two signals, one question, and they are **not** the same claim:

* the learned signal claims to rank cases by *how likely the classifier is to be
  wrong*, which is only indirectly about the reader;
* the certificate claims the *photograph does not carry the information*, which
  is a statement about the image a reader looks at, and so should transfer to a
  human reader if it is true at all.

The certificate is therefore the more falsifiable of the two, and it is the one
this study is really aimed at.

**Falsification, committed in advance.** If the certificate score's weighted AUC
against reader referral does not clear 0.5 by more than its 95% cluster
bootstrap interval, the physics track is measuring something readers do not act
on. The response is to drop the second-reader framing from the paper, not to
soften it into "correlates weakly". The same rule applies to each learned
signal independently.

## 2. The instrument

Two questions per film, because the system makes two different decisions and
collapsing them would make the result uninterpretable:

**Q1 (referral).** *"Reading this film as presented, would you seek a second
opinion before reporting it?"* — 1 no, confident single read / 2 probably not /
3 borderline / 4 probably yes / 5 I would not report this alone.
Binarised at **>= 4**, fixed in advance (`REFER_BINARY_CUT`).

**Q2 (adequacy).** *"Is the image quality adequate to report, or would you ask
for the film to be re-photographed?"* — adequate / borderline / inadequate.

**Q3 (reason).** One code from `blur, glare, shadow, contrast, cropping,
artefact, ambiguous_finding, subtle_finding, other`, plus free text. The codes
mirror `physics/triage.py`'s reason vocabulary so the two can be cross-tabulated
directly.

Q1 versus Q2 is the clinically meaningful split and the same one the triage
module already makes: *the finding is ambiguous* (refer to a specialist) versus
*the photograph is inadequate* (retake it). A system that gets the overall flag
rate right while systematically confusing the two is worse than useless in a
clinic, because it sends films to a specialist that a health worker could have
fixed with a second photograph.

Readers see a PNG named by an opaque case id and nothing else: no clinic, no
severity, no label, no model output. `scripts/reader_study.py design
--export-images` regenerates exactly the photographs the rows were scored on
(the capture seed is CRC32 over (path, severity, seed), so it reproduces bit for
bit in a later process) and names them by case id.

## 3. Sampling, and why it is not random

Random sampling would spend the reader budget on the bulk of the corpus, where
both signals agree and the answer is already known. The question lives in the
**discordant** cells: the certificate says the photograph cannot carry the
finding while the classifier is confident, or the reverse.

So the sample is **balanced over the joint grid** of physics-margin tertile ×
learned-uncertainty tertile, plus a dedicated stratum for photographs where the
certificate **abstains** (no fiducials detected — 23.5% of this corpus, and a
category in its own right: the study asks whether the certificate's abstentions
land on films a reader also calls inadequate).

Balance makes the sample unrepresentative on purpose, so every drawn row carries
`sampling_weight = population(cell) / drawn(cell)` and **every estimator in the
module is weighted**. The unweighted numbers would be design artifacts.

Two further details that are not cosmetic:

* **One photograph per film per reader.** The corpus has each film at five
  severities. Showing a reader the same film twice at different severities leaks
  the answer and inflates agreement, so a film is spent on at most one cell in
  the whole sample. That makes allocation a degree-constrained bipartite
  matching rather than a per-cell draw, and it has to be solved as one: greedy
  cell-by-cell allocation lets early cells spend films that a later, scarcer
  cell was the only one able to use, which on this corpus produced cells of 10
  beside cells of 2 — a "balanced" design underpowered exactly in the discordant
  cells it exists to buy. Augmenting paths, filled one level at a time, give
  12 cells × 10 distinct films.
* **10% of films are shown twice**, in shuffled positions. Intra-reader
  repeatability then comes out of the study itself. Without it, a low
  model-versus-reader agreement has two explanations — a bad model, or a reader
  who does not agree with themselves — and no way to separate them.

Bootstrapping resamples **films**, not photographs, for the same reason.

## 4. The analysis, fixed before the data exists

With four signals, two questions and twelve strata there are more than enough
contrasts to find something. The read-out is therefore fixed:

| Block | Quantity | Why this one |
| --- | --- | --- |
| Signals | weighted AUC + 95% cluster-bootstrap CI, per signal, against Q1 | the headline; `physics_score` is the pre-registered primary |
| Policy | agreement, Cohen's kappa **and** Gwet's AC1 between the deployed defer set and the reader referral set | referral is rare, and kappa collapses under skewed marginals — the first-kind paradox. Where they disagree, believe the AC1 |
| Policy | the two error directions kept apart | deferring a film no reader would refer costs clinic time; answering one every reader would refer is the failure that matters |
| Triage | kappa/AC1 between `triage_action == retake` and Q2 "inadequate" | *which action*, not just whether something is wrong |
| Ceiling | inter-reader ICC(2,1), pairwise kappa/AC1, and the oracle AUC under that noise | see below |

**The ceiling is part of the result, not a caveat.** Radiologists do not agree
with each other about second reads — single-reader reliability on chest-film
calls is routinely reported in the 0.4–0.6 band — and that caps what *any* model
can score against one reader's ratings. `reader_noise_ceiling` measures the cap
by handing an oracle the latent referral propensity itself and scoring it
against the same noisy labels the model will face. A measured AUC of 0.72
against a ceiling of 0.78 is a nearly-solved problem; reported without the
ceiling it looks mediocre. The study reports both, always.

## 5. What the rig already measured, before any radiologist

Run on the 600 real test rows from `scripts/physics_deferral_real.py`
(120 Shenzhen/Montgomery films × 5 severities, one checkpoint's calibrated
probabilities joined to certificates computed on the same capture):

**5.1 The two signals are close to orthogonal — so the study is worth running.**
Spearman rank correlation between the certificate score and MC-dropout spread is
**−0.10**. At a matched flag rate of 0.40, a quarter of photographs are flagged
by the physics alone and a quarter by the learned signal alone. If they agreed,
no reader study could separate them; they do not, so the discordant cells are
populated and the design has something to measure.

**5.2 Two of the binary decisions are constant on this corpus, and one contrast
is therefore unmeasurable as things stand.** On these 600 rows
`triage_action == "retake"` for **all** of them and `model_confident` is True for
**all** of them. The deployed policy defers nothing and asks for a retake on
everything. A kappa against a constant is undefined, so the **policy-agreement
block cannot run at this operating point** — not because of anything about
readers, but because the operating point has no variation to agree about. The
sampler works on the continuous margin instead, which is why it is stratified on
tertiles rather than on the verdict. Fixing this is a prerequisite for the study,
not a footnote to it: either the deferral threshold has to be set somewhere the
model actually defers, or the triage rule has to be evaluated at a severity range
where it does not saturate.

**5.3 The worst-finding verdict is saturated; the per-finding verdicts are not.**
Every one of the 459 certified photographs is INSUFFICIENT for a miliary nodule
(margin at most −0.24 dB), so the headline verdict carries no information here.
Per finding it is a different picture — infiltrate splits 188 detectable / 88
marginal / 183 insufficient, cavity wall 59 / 67 / 333, consolidation 449 / 6 / 4.
The study's secondary analyses use the **infiltrate** verdict, the one with an
operating point that is not degenerate.

`docs/figures/run1/reader_study_design.json` carries all of these numbers.

## 6. Power, and how many readers

`scripts/reader_study.py power` simulates the whole pre-registered test — the
generative reader panel, the balanced sample, the cluster bootstrap, the
decision rule — over a grid of n, reader count, single-reader ICC and effect
size. The type-I rate at the null is reported next to every power number, because
a bootstrap interval on a weighted AUC is not guaranteed to be calibrated and
should be checked rather than assumed. Numbers in
`docs/figures/run1/reader_study_power.csv`.

The reader model is deliberately not fitted to anything in this project: ICC and
referral prevalence come from published ranges, and `signal_r` — the correlation
between the model signal and the latent referral propensity — *is* the effect
size being powered for.

At the central setting (single-reader ICC 0.5, referral prevalence 0.25):

| n films | readers | r = 0.30 | r = 0.45 | r = 0.60 | AUC ceiling |
| --- | --- | --- | --- | --- | --- |
| 60 | 1 | 0.20 | 0.41 | 0.70 | 0.846 |
| 120 | 1 | 0.39 | 0.73 | 0.93 | 0.846 |
| 240 | 1 | 0.70 | 0.96 | 1.00 | 0.846 |
| 60 | 3 | 0.27 | 0.52 | 0.84 | 0.913 |
| 120 | 3 | 0.52 | **0.85** | 0.99 | 0.913 |
| 240 | 3 | 0.78 | 0.98 | 1.00 | 0.913 |

Three things follow, and they are the operational conclusions:

* **n = 120 films, 3 readers** is the design: 85% power against a moderate
  effect (r = 0.45), which is the smallest effect worth calling a result. That
  is 132 readings per reader once the 10% repeats are included — roughly one
  sitting.
* **A third reader buys more than doubling the films.** Going 1 → 3 readers at
  n = 120 moves power from 0.73 to 0.85 *and* lifts the ceiling from 0.846 to
  0.913, because consensus reliability rises from 0.50 to 0.75 (Spearman-Brown).
  Doubling films to n = 240 with one reader reaches similar power but leaves the
  ceiling where it was. Reader time is the cheaper axis here.
* **The ceiling is 0.85–0.91, not 1.0.** Against a single reader at ICC 0.5, an
  oracle that knows the referral propensity exactly scores 0.846. Any measured
  AUC has to be read against that number.

Type-I rate at the null across the whole grid: **0.023–0.050** against a nominal
0.025, so the bootstrap decision rule is honest at these sample sizes — checked,
not assumed.

## 7. Running it

```bash
# draw the sample, emit the blinded sheet + key + the design's own numbers
python scripts/reader_study.py design --rows outputs/physics_deferral_rows.csv \
    --n-cases 120 --export-images

# the ceiling and the power grid; run before booking anyone
python scripts/reader_study.py power --out outputs/reader_study_power.csv

# when the ratings come back, one CSV per reader
python scripts/reader_study.py analyze --key outputs/reader_study_key.csv \
    --ratings ratings_r1.csv ratings_r2.csv ratings_r3.csv
```

`analyze` refuses to run on partial ratings: every drawn case must be rated by
every reader, or the inter-reader block silently becomes a different estimand.

## 8. What this still does not close

* The films are **simulated captures** of archive scans, not real phone photos of
  real films. A reader rating these is rating this project's forward model as
  much as the underlying image; `docs/REAL_RECAPTURE.md` is the other half.
* Ratings are of images in isolation, with no clinical history and no priors,
  which is not how a second read is actually requested.
* Agreement with reader referral is still not evidence about **patient
  outcomes** — see `docs/DEPLOYMENT_CHECKLIST.md` §E, which keeps that as a
  separate open item on purpose.
