# Limitations

Written proactively, per the Phase 5 plan, so the paper's limitations section is
drafted before results exist and cannot be quietly trimmed to fit a nicer story.
Each item says what the limitation is, why it matters, and what would close it.

## 1. No result in this repository has been produced by a real training run

Every number currently reachable — accuracy, ECE, AURC, coverage — comes from
synthetic data in `scripts/smoke_test.py` or from untrained/2-epoch checkpoints.
The pipeline is verified end to end; the *findings* are not. Nothing in this repo
should be quoted as an empirical claim until `scripts/run_experiments.py` has been
run on the real cohorts on a GPU.

**Closes when:** the LOCO sweep runs on real data and `outputs/loco_sweep/` is
populated.

## 2. All degradation is synthetic; there are no real phone recaptures

`data/degradation.py` is a physics-style simulation and
`data/degradation_learned.py` is an unpaired learned generator. Neither has been
validated against actual photographs of actual films, because no such set exists
here. `scripts/ablate_degradation.py` is built to score them against real
recaptures and currently can only score them against each other, which cannot
establish realism — only disagreement.

This is the single largest threat to external validity. A model robust to *our*
blur/glare model is not thereby robust to a Redmi 9A in a Ugandan clinic.

**Closes when:** the pilot protocol in `data/real_recapture/README.md` is
executed (20–30 images is enough), or a public real-recapture set (CheXphoto real
subset, PhysioNet `cxr-phone`) is licensed and used as the reference distribution.

## 3. Two held-out clinics, not four

Four sources contribute data, but NIAID and Belarus are TB-only and RSNA is
normal-only, so only Montgomery (138 images) and Shenzhen (662) can serve as
two-class holdout folds. `data/splits.py` refuses the others by default.

Consequences: the cross-site claim rests on two rotations, Montgomery's fold has
138 test images so its confidence intervals are wide, and "leave-one-clinic-out
across four clinics" would be a misstatement. Report per-fold n and interval
estimates, not point accuracies alone.

**Closes when:** additional two-class sources are added.

The other half of that sentence used to read "balanced multi-class cohorts are
constructed (mixing RSNA normals with NIAID positives under the DUA)". That
route is now measured and it is closed. `scripts/audit_clinics.py confound` fits
a logistic regression on nine low-level capture statistics -- brightness,
contrast, dynamic range, entropy, sharpness, pixel dimensions, nothing that can
see anatomy -- and on a hybrid cohort built from one source's normals and
another's positives it calls the label at **AUC 1.000**. The obvious objection,
that it rides on image dimensions the loader resamples away, was tested: at
224 px it is still 1.000, carried by mean brightness instead. A hybrid cohort is
not a clinic, and a classifier's accuracy on one says nothing about TB.

Two further numbers from the same audit belong here. The real folds are not
innocent either -- Shenzhen scores 0.853 on that test (0.775 after resampling)
and Montgomery 0.647 (0.672) -- so a capture-level shortcut is available inside
a genuine two-class fold, well short of the hybrid's 1.000 but well above
chance. That is an argument for more sites, not for more modelling.

`docs/SOURCES.md` surveys what those sites could be. The near-term answer is the
NITRD DA and DB sets (India, ~150 images each, both classes, no access gate, and
shot on two different machines at one institute), which would take the rotation
from two folds to four.

## 4. The uncertainty target is a proxy for correctness, not radiologist agreement

`manifest.uncertainty_target_from_severity` derives the weak supervision label
monotonically from applied degradation severity. That encodes "a degraded image
deserves low confidence," which is a reasonable prior and is what the
channel-capacity framing argues for — but it is not the same as "a radiologist
would want a second read here." The second-reader framing in
`docs/phase1_framing.md` is a motivating analogy, not a measured quantity. No
clinician has labelled anything in this project.

**Closes when:** a radiologist rates a sample of held-out images for
"would you seek a second opinion," and that is correlated against predicted
uncertainty.

**Status: everything except the radiologist is now built and run.**
`docs/READER_STUDY.md` pre-registers the protocol, the instrument and the
falsification criterion; `eval/reader_study.py` + `scripts/reader_study.py` draw
the balanced sample (12 cells x 10 films over the margin x uncertainty grid,
including a stratum for the certificate's abstentions), export the blinded
photographs, and run the weighted analysis. Three things it already measured
without a reader:

* the two signals are near-orthogonal on the real corpus (Spearman -0.10), so
  the study can separate them -- if they had agreed, no reader study could;
* **the policy-agreement contrast cannot run at the current operating point**:
  on all 600 test rows `triage_action` is `retake` and `model_confident` is
  True, both constant, and a kappa against a constant is undefined. This is a
  prerequisite to fix, not a footnote;
* the reader-noise ceiling is 0.85 (one reader) to 0.91 (three), not 1.0, so
  n = 120 films x 3 readers gives 85% power against a moderate effect.

The limitation itself is unchanged until a clinician actually rates the sample.

## 5. Deferral assumes a human is available to defer *to*

The entire safety argument is that low-confidence cases go to a person. In the
rural, cold-chain-free settings this project targets, that person may not exist,
may not be reachable offline, or may be the same overloaded health worker who
took the photo. A high deferral rate is not free: it is a workload transfer, and
at some rate it makes the system useless rather than safe.

`human_rescue_rate` measures whether deferred cases were *worth* deferring, which
is the right question, but it cannot measure whether the referral pathway exists.

**Closes when:** deferral rate is budgeted against a stated staffing assumption —
see `docs/DEPLOYMENT_CHECKLIST.md`.

## 6. The conformal guarantee does not formally transfer across clinics

Split-conformal coverage assumes calibration and test data are exchangeable.
Under LOCO they are not, by construction. `eval/conformal.py` reports the
shortfall between guaranteed and achieved coverage as a diagnostic rather than
claiming the guarantee holds on the held-out clinic. Do not state a 1-α guarantee
on a new site.

**Closes when:** it does not — this is inherent. It can only be reported honestly.

## 7. TB-Net is reimplemented, not reproduced

`models/tbnet.py` is reproduction path B: an attention-condenser-flavoured CNN
tuned to TB-Net's reported ~4.24M parameters, not a port of the released
TensorFlow 1.15 checkpoint. Its MACs are roughly 2× TB-Net's at matched
parameters, documented in the module. Any comparison to TB-Net's published 99.86%
is a comparison to a *reimplementation*, and must say so.

**Closes when:** path A (porting the TF checkpoint layer by layer and verifying
activations on a fixed input) is done, or the comparison is dropped.

## 8. Retrospective, public, curated data

All cohorts are curated research datasets of already-diagnosed patients. They
under-represent the ambiguous, early-stage, and comorbid presentations that
dominate real screening, and their normals are not drawn from a screening
population. Reported sensitivity/specificity will be optimistic relative to
field deployment regardless of how carefully the splits are done.

## 9. Not a clinical device

This is a research tool. It is not validated, not regulated, not approved, and
not for diagnosis. See `docs/DEPLOYMENT_CHECKLIST.md` for what would have to be
true before any clinical use.

## 10. The falsification test currently fails, in the unsafe direction

`python scripts/validate_physics.py` (full, not `--quick`), run 2026-08-15: median
predicted/empirical detectability ratio is **0.50** (IQR 0.12–1.90), with only 31% of the 16
severity×finding conditions individually passing. `ratio < 1` means the certificate can pass a
photograph an optimal detector cannot actually read — the dangerous direction, and the opposite
of the ≈1.7 previously stated in `docs/PHYSICS.md`, which came from an unrepresentative
`--quick` run. **The certificate must not be deployed as a safety valve until this is fixed.**

The channel-recovery numbers point to why: veil-fraction estimation error is small while
fiducial coverage is `full` (≈−0.01) and collapses to ≈−0.73 once severity degrades coverage to
`none`, with `density_abs_rmse` more than doubling over the same range. The blind estimator has
no reliable fallback for the glare field once the beam-stop rim that measures it is lost, so the
reported floor stays too small exactly when the photo is worst. This is broad, not narrow: by
severity 0.75, three of the four findings sit far below 1 (`miliary_nodule` 0.07, `cavity_wall`
0.12, `consolidation` 0.12), and for `cavity_wall`/`consolidation` the underlying linear fit is
still good (R² = 0.96, 0.92) there — so this is a real miscalibration, not test noise. The test's
own contrast-probe bracket, centred on the model's own predicted floor
(`physics/validate.py:308-315`), is a separate, narrower problem: it specifically wrecks the fit
(R² < 0.7) only for `miliary_nodule` at severity 0.5 and `cavity_wall` at severity 0.5, and the
exact ratios at those two cells shouldn't be quoted at face value. See `docs/PHYSICS.md` §3 for
the full per-condition breakdown.

**Closes when:** the channel estimator gets a real fallback for lost beam-stop coverage (or the
certificate abstains earlier, before coverage fully drops, rather than continuing to assert a
floor computed from a degraded fit), and `detectability_experiment`'s probe bracket is decoupled
from its own prediction so the reported ratio isn't self-referential.

## 11. The fiducial coverage audit likely overstates real coverage

`scripts/audit_fiducials.py` reported 32.1% certifiable (full+partial) on the aggregated Kaggle
corpus (`outputs/fiducial_audit.csv/.json`), and that number decided the real-photo-vs-simulated
path (§ "Decide the path", below 50% → simulated re-photography). Visual QA of the detections
(`outputs/figures/11_fiducials_real.png` and a 9-image ad hoc sample, 2026-08-15) found the
collimation-border/beam-stop detector systematically false-positive on ordinary chest anatomy:
`detect_collimation` (`physics/fiducials.py:245-308`) accepts any image where the outer 3%
border strip is ≥0.12 brighter than the darkest 5% of the centre, which lung fields satisfy on
their own with no real film border present. 9/9 sampled `partial`-coverage images showed the
detected "beam stop" landing on lung tissue, not a genuine direct-exposure region.

The qualitative verdict (simulated re-photography path) is not in question — a lower true
coverage rate only reinforces it — but the 32.1% figure, and the `partial` bucket specifically,
should be reported as a likely overestimate rather than a clean measurement until the detector
is hardened and re-validated.

**Closes when:** `detect_collimation`/`detect_beamstop` require the bright border to be
low-variance/uniform (not merely brighter than the darkest interior pixels) and are re-validated
against images with known ground truth, then the audit is re-run for a trustworthy number.
