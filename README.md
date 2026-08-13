# TB-Trust

Trustworthy tuberculosis screening from chest X-rays when the image is a phone
photo of an analog film, not a clean scan from a dedicated machine. Three things
have to work together: robustness to smartphone-capture artifacts, calibrated
uncertainty that knows when a photo is too degraded to trust, and a deferral
policy that says "retake the photo / refer to a specialist" instead of quietly
returning a wrong answer. Evaluation is leave-one-clinic-out, so cross-site domain
shift and capture degradation are measured at the same time.

**On "four clinics", precisely.** The manifest draws on four public sources
(Montgomery, Shenzhen, NIAID, RSNA), but only Montgomery and Shenzhen contain
both classes, so only those two can serve as held-out folds — sensitivity or
specificity is undefined on a single-class test set, and there the clinic label
is a near-perfect proxy for the diagnosis. `data/splits.py` enforces this by
default. So the honest phrasing, and the one the paper must use, is: **four
sources contribute training data; the rotation reports two two-class holdout
folds.** Anything stronger will not survive review. See the provenance note below.

This repo is the starting codebase. The plumbing runs end to end today on a
placeholder model; the research contributions have clearly marked slots to fill.
No accuracy number in this repo has been produced by a real training run yet.

## What's already here (and tested)

- **Smartphone degradation pipeline** (`data/degradation.py`): motion/defocus blur,
  lightbox glare, uneven shadow, capture angle, JPEG artifacts, resolution loss.
  Continuous severity, and every image carries a record of what was applied so you
  can derive the weak uncertainty labels.
- **Learned degradation + ablation** (`data/degradation_learned.py`,
  `scripts/ablate_degradation.py`): a small unpaired adversarial generator as a
  second degradation strategy, and a script that scores any strategy's realism
  against real phone recaptures (or against each other, until real recaptures
  exist -- see `data/real_recapture/README.md`).
- **Leave-one-clinic-out splits** (`data/splits.py`) with a guard that refuses
  single-class holdouts (see the provenance note below, it will bite you otherwise).
- **Evaluation suite**: calibration (ECE/MCE, reliability diagram, and temperature
  scaling actually *applied* — fitted on val, never re-fit on test), safe-deferral
  / risk-coverage curve with a threshold tuned on **validation** and applied (not
  re-tuned) on test, split-conformal coverage guarantee (`eval/conformal.py`),
  Brier score + Brier Skill Score + Murphy decomposition
  (`eval/forecast_verification.py`), human-rescue rate, per-clinic cross-site
  analysis, and a degradation-vs-uncertainty check
  (`eval/degradation_uncertainty.py`) that asks whether each uncertainty signal
  actually rises with capture degradation -- the premise the "retake the photo"
  message depends on, and not the same question as a good AURC.
- **Uncertainty methods compared head to head** (`eval/run.py`): every method
  scores the *same* temperature-scaled probabilities and differs only in the
  signal the deferral policy ranks on — `confidence` (max(p, 1-p), the baseline),
  `mc_dropout` (predictive spread), `head` (the learned uncertainty head, or
  evidential vacuity), and `ensemble` (member disagreement). AURC per method at
  every degradation severity is the comparison table the paper needs. Holding the
  point prediction fixed is deliberate: it isolates ranking quality instead of
  confounding it with a change in the classifier's own predictions.
- **Full LOCO sweep aggregation** (`scripts/run_experiments.py`): trains/evaluates
  every fold and turns the results into the actual cross-site comparison
  (generalization-gap table, per-clinic calibration heatmap, forecast-verification
  metrics per clinic), not just per-fold JSONs to eyeball individually.
- **Runnable model + training/eval loop** driven by YAML configs, with
  `model.arch: baseline | tbnet | evidential` all sharing one training/eval path.
- **Uncertainty methods** (`models/uncertainty.py`, `models/evidential.py`,
  `models/ensemble.py`): MC-dropout, evidential deep learning (the featured
  calibration-focused head, one forward pass), and deep ensembles (via
  `scripts/train_ensemble.py`) -- comparable head to head on the same held-out
  clinic. `eval/sequential_deferral.py` adds an adaptive-stopping MC-dropout rule
  and a CUSUM chart for detecting per-clinic drift over a stream of images.
- **Domain generalization** (`losses/dg.py`, `models/grl.py`,
  `models/clinic_film.py`): Deep CORAL, IRM, DANN (domain head behind a
  gradient-reversal layer) and clinic-conditional FiLM, selected with
  `dg.method` / `model.clinic_film` in any config. Deferral *absorbs* the
  cross-site gap at inference; these attack it during training, and the two are
  orthogonal -- any `dg.method` combines with any uncertainty method, so
  `configs/loco_montgomery{,_coral,_dann,_film}.yaml` is a four-way ablation on
  one fold. `scripts/clinic_stats.py` measures the shift itself (per-clinic
  resolution, brightness, contrast), so "the clinics differ" is a number rather
  than an assumption.
- **Efficiency + robustness checks** (`scripts/benchmark_efficiency.py`,
  `eval/adversarial_degradation.py`): params/MACs/CPU latency for every
  architecture, and a worst-of-N-query black-box degradation search that checks
  whether predicted uncertainty actually rises on the images it makes harder.
- A **smoke test** that verifies all of the above with no data and no GPU.

## Quickstart

```bash
git clone <your-fork-url> tb-trust && cd tb-trust
python -m venv .venv && source .venv/bin/activate      # or: conda env create -f environment.yml
pip install -e .                                        # core deps + console scripts
python scripts/smoke_test.py                            # should print all PASS
```

`smoke_test.py` runs the whole core on synthetic images, so a green run means the
install is good before you touch real data.

Then get data and train:

```bash
# 1. data (see scripts/download_data.py for the per-source access rules)
python scripts/download_data.py --kaggle-aggregated      # fastest start
python scripts/download_data.py --print-instructions     # NLM / RSNA / NIAID steps

# 2. build the manifest and read the class balance
python scripts/build_manifest.py --raw data/raw --out data/processed/manifest.csv

# 3. train a leave-one-clinic-out run (Montgomery held out) and evaluate it
tbtrust-train --config configs/loco_montgomery.yaml
tbtrust-eval  --config configs/loco_montgomery.yaml --checkpoint outputs/montgomery/best.ckpt
```

Overrides are `key.subkey=value` on the command line, e.g.
`tbtrust-train --config configs/loco_montgomery.yaml model.backbone=resnet50 train.epochs=30`.

## Two facts to read before you design experiments

**TB-Net's released code is TensorFlow 1.15 / checkpoint format**
(github.com/darwinai/TuberculosisNet, paper arXiv:2104.03165). Rather than drag
the team into a Python 3.6 / TF1 environment, the default runnable model is a
DenseNet-121 (CheXNet-style) baseline in PyTorch. Reproducing TB-Net's
attention-condenser design is a task with its own slot in `models/tbnet.py`, and it
drops into the same training loop. Treat any TB-Net *accuracy* numbers as
unvalidated until real training runs happen -- but the params/MACs axis has
already been measured (`scripts/benchmark_efficiency.py`): `tbnet.py`'s widths
are tuned to TB-Net's reported ~4.24M params, and it costs about 2x TB-Net's
0.42 GMACs at that same param count, a real architecture-efficiency gap, not
just an unverified claim.

**The aggregated Kaggle TB dataset mixes sources with skewed class balance.**
Montgomery and Shenzhen (NLM) have both normal and TB. NIAID and Belarus are
TB-only; RSNA contributes normals only. So holding out NIAID or RSNA alone gives a
test set with a single class, where sensitivity or specificity is undefined.
Montgomery and Shenzhen are your clean two-class holdouts. The split code enforces
this by default; `configs/loco_montgomery.yaml` and `loco_shenzhen.yaml` are the
folds that "just work."

## Repo layout

```
configs/            yaml experiments (default.yaml + one file per run)
scripts/            download_data, build_manifest, clinic_stats, smoke_test,
                    train, evaluate, train_ensemble, benchmark_efficiency,
                    evaluate_adversarial_robustness, ablate_degradation,
                    run_experiments
src/tbtrust/
  data/             degradation pipeline, manifest + provenance, LOCO splits,
                    torch Dataset, per-clinic shift statistics
  models/           baseline, tbnet reproduction, evidential head, deep ensemble,
                    MC-dropout, gradient reversal, clinic-conditional FiLM
  losses/           domain-generalization objectives (CORAL, IRM)
  train/            training loop + CLI (arch-dispatched: baseline/tbnet/evidential,
                    with the DG penalty layered on top)
  eval/             calibration, conformal, deferral, sequential/CUSUM deferral,
                    adversarial degradation, cross-site, forecast verification,
                    degradation-vs-uncertainty correlation, metrics, eval CLI
  utils/            seeding, io
tests/              pytest smoke tests + domain-generalization tests
data/               raw/, processed/, and real_recapture/ (all gitignored except real_recapture/README.md)
```

`ONBOARDING.md` maps the five project phases to these files and suggests how to
split the work across people.

## Notes

- Every metric the paper reports is computed by the local implementations in
  `eval/` — calibration (ECE/MCE), risk-coverage/AURC, temperature scaling,
  split-conformal coverage, and the forecast-verification suite. There is no
  third-party metrics dependency. `torch-uncertainty` covers similar ground and
  is installable as the optional `[uq]` extra (`pip install -e ".[uq]"`) if you
  want to cross-check against a maintained reference, but nothing imports it.
- Everything under `data/` and `outputs/` is gitignored. Don't commit images or
  checkpoints.
- This is a research tool, not a clinical device. It is not for diagnosis.
