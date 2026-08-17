# Run 1: the first measured numbers in this repo

Every number below was measured, not estimated. Corpus is the raw NLM Montgomery
and Shenzhen sets (800 images, both clinics two-class), pulled from
`openi.nlm.nih.gov` and mounted on Kaggle. Training and evaluation ran on a
Tesla T4. The physics track ran on CPU.

Where a number is an estimate it says so explicitly.

## 1. Fiducial coverage — the gate, and it is negative

`scripts/audit_fiducials.py`, 800 images, full resolution.

| clinic | n | full | partial | none | marker | usable edge |
|---|---|---|---|---|---|---|
| montgomery | 138 | 0.09 | 0.10 | 0.81 | 0.90 | 0.33 |
| shenzhen | 662 | 0.00 | 0.03 | 0.97 | 0.57 | 0.17 |
| **ALL** | **800** | **0.01** | **0.04** | **0.94** | **0.63** | **0.20** |

**Certifiable (coverage != none): 5.5%.**

The lead marker survives digitisation reasonably (0.90 Montgomery, 0.57
Shenzhen). The collimation border and direct-exposure rim do not: 94% of the
corpus has no optical beam stop, and Shenzhen has *zero* images with full
coverage.

**Consequence.** The real-photo path is not available on these archives. The
physics track is a claim about simulated re-photography (`physics/film.py`
painting fiducials back on), or about a prospective capture protocol where the
fiducials are controlled at acquisition. It is not a claim about inverting
archived radiographs, and the paper must say so with this number.

This also promotes `data/real_recapture/` from "highest-leverage next step" to
the only external validation available to the physics track.

## 2. Cross-site generalisation gap

Leave-one-clinic-out, DenseNet121, 20 epochs, images pre-resized to 224 px.

| fold | val accuracy | held-out accuracy | drop |
|---|---|---|---|
| Montgomery held out | 0.919 | **0.717** | −20 pts |
| Shenzhen held out | 0.952 | **0.625** | −33 pts |

At severity 0.0, on the held-out clinic:

| fold | n | accuracy | sensitivity | specificity | Brier | ECE | MCE |
|---|---|---|---|---|---|---|---|
| Montgomery | 138 | 0.717 | **0.466** | 0.900 | 0.182 | 0.131 | 0.575 |
| Shenzhen | 662 | 0.625 | **0.295** | 0.966 | 0.235 | 0.123 | 0.240 |

The sensitivity collapse is the result that matters clinically and the one a
headline accuracy figure hides. On an unseen clinic the model misses 53% and 71%
of TB cases while holding specificity at 0.90–0.97 — it has largely learned to
answer "normal".

### Accuracy under degradation

| severity | Montgomery held out | Shenzhen held out |
|---|---|---|
| 0.00 | 0.717 | 0.625 |
| 0.25 | 0.674 | **0.675** |
| 0.50 | 0.681 | 0.644 |
| 0.75 | 0.652 | 0.600 |
| 1.00 | 0.645 | 0.583 |

Note the Shenzhen fold *improves* from severity 0.0 to 0.25 (accuracy
0.625 → 0.675, sensitivity 0.295 → 0.530). Training randomised severity over
[0, 1], so a pristine image is out-of-distribution relative to the training
mean. This is a finding about the augmentation scheme, not noise.

## 3. Uncertainty methods, head to head

AURC (lower is better), identical temperature-scaled probabilities, threshold
tuned on val and applied to the held-out clinic.

| method | Montgomery @0.0 | Montgomery @0.5 | Shenzhen @0.0 | Shenzhen @0.5 |
|---|---|---|---|---|
| confidence | 0.141 | 0.155 | 0.280 | **0.259** |
| mc_dropout | **0.138** | **0.154** | **0.271** | 0.291 |
| head | 0.241 | 0.347 | 0.331 | 0.328 |

**The learned uncertainty head loses to plain max-probability confidence** on
every fold and every severity, by more than 2× on Montgomery. Its correlation
with true degradation severity is `spearman_rho = 0.0` on the Shenzhen fold
despite being trained on a degradation-derived target; `confidence` and
`mc_dropout` both reach 0.9.

### Temperature scaling

| fold | T | ECE before | ECE after |
|---|---|---|---|
| Montgomery | 1.519 | 0.078 | **0.089** |
| Shenzhen | 0.598 | 0.159 | 0.138 |

Temperature scaling made calibration *worse* on the Montgomery fold. The
Shenzhen fold fitted T < 1, i.e. sharpening rather than smoothing.

## 4. Conformal prediction fails under domain shift

Split conformal, alpha = 0.1 (target coverage 0.90).

| fold | n_cal | in-dist coverage | held-out coverage | shortfall | abstention |
|---|---|---|---|---|---|
| Montgomery | 49 | 0.980 | **0.804** | 0.096 | 0.254 |
| Shenzhen | 10 | 1.000 | 0.947 | −0.047 | **0.739** |

The Montgomery fold's 90% guarantee delivers 80.4% on the held-out clinic — a
17.6-point coverage drop from in-distribution. The Shenzhen fold holds coverage
only by abstaining on 73.9% of cases. Its calibration set is n=10, so treat that
fold as indicative only.

## 5. Physics certificates at 1024 px

`scripts/physics_certificates.py --size 1024 --limit 100`, 500 rows, simulated
re-photography (the real-photo path is unavailable — see §1).

| severity | abstain | insufficient | marginal | detectable | margin_db | retake | refer |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 1.00 | 0.00 | **0.00** | −10.0 | 1.00 | 0.00 |
| 0.25 | 0.17 | 0.82 | 0.01 | 0.00 | −10.2 | 1.00 | 0.00 |
| 0.50 | 0.29 | 0.70 | 0.01 | 0.00 | −10.6 | 1.00 | 0.00 |
| 0.75 | 0.31 | 0.69 | 0.00 | 0.00 | −11.3 | 1.00 | 0.00 |
| 1.00 | 0.29 | 0.71 | 0.00 | 0.00 | −12.8 | 1.00 | 0.00 |

`detectable = 0.00` at **every** severity including zero, and `retake = 1.00`
everywhere. The certificate declares a pristine synthetic capture insufficient
and instructs a retake on 100% of photographs. As a triage gate it currently
carries no information: it fires identically on a perfect image and a ruined
one, and the margin moves only 2.8 dB across the whole severity range.

The retake/refer split is degenerate for the same reason — nothing is ever
referred, because nothing is ever certified as readable.

## 6. Per-clinic domain shift

`scripts/clinic_stats.py --sample 100`.

| clinic | megapixels | mean brightness | mean contrast |
|---|---|---|---|
| montgomery | 19.67 | 0.440 | 0.326 |
| shenzhen | 8.38 | 0.620 | 0.248 |

A 2.3× resolution difference and clearly separated brightness. This is the
concrete domain-shift table for the fairness/audit section.

## 7. Measured compute

Replaces the estimates in the roadmap, all of which were guesses.

| step | measured |
|---|---|
| pre-resize 800 images to 224 px | 116 s (0.145 s/image) |
| LOCO training, 20 epochs, T4 | minutes per fold, not 30–60 |
| physics certificates, 1024 px | ~1.9 s/image × 5 severities (unchanged) |

Two hardware notes. Kaggle's **P100 no longer works**: it is compute capability
sm_60 and the current PyTorch build supports sm_70 and above, failing with
`no kernel image is available for execution on the device`. Use T4. And
`data/dataset.py` re-decodes full-resolution PNGs on every `__getitem__`, so
without a pre-resize pass training pays that 116 s every epoch.

## 8. A defect in the falsification gate itself

`scripts/validate_physics.py` filtered detectability conditions on `ratio > 0`
before taking the median. Conditions where the matched filter achieves zero
discrimination produce `empirical = inf` and therefore `ratio = 0`, so they were
silently dropped — and those are precisely the estimator's worst failures.

Measured on the retired NOMINAL table:

| | seed 0 | seed 1 |
|---|---|---|
| conditions entering the median | 16 / 16 | **12 / 16** |
| reported median ratio | 0.45 | 1.81 |
| max predicted floor | 0.62 OD | **10.53 OD** |

The run that failed harder reported the safer-looking number. A floor of 10.53
OD is not a statement about a photograph — developed chest film spans roughly OD
0.2–3.0 — it is the inversion having diverged.

The 4× swing in the headline calibration ratio is this censoring, not seed
variance. No single run's median could be quoted as the bound's calibration.

The accompanying patch adds `--seeds` (default 5), classifies conditions as
`degenerate` (zero discrimination) or `unphysical` (floor above 4.0 OD) instead
of dropping them, reports per-seed medians and the across-seed spread, and fails
the gate on instability or censoring rather than only on the point estimate.

## Status by roadmap box

| box | state |
|---|---|
| 1. Restore the data | partial — Montgomery + Shenzhen (the only two-class clinics) |
| 2. Fiducial coverage audit | done — **5.5% certifiable** |
| 3. Decide real-photo vs simulated | decided — simulated, forced by box 2 |
| 6. LOCO folds | done — DG ablation still blocked, see below |
| 7. `tbtrust-eval` per checkpoint | done |
| 8. Certificates at 1024 px | done — certificate is degenerate |
| 13. Per-clinic calibration table | data done, prose outstanding |

The four-way DG ablation (none/coral/dann/film) remains blocked: with two
clinics, holding one out leaves a single training domain, and CORAL and DANN
both operate by aligning representations *across* source domains. Those arms
would differ from `none` only by noise. It needs a third two-class clinic.
