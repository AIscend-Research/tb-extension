# xctb — cross-cohort, uncertainty-aware TB chest X-ray triage

A lightweight chest X-ray TB screener that is honest about deployment. It is
trained and tested with strict leave-one-cohort-out splits (train on some
clinics, test on a clinic the model has never seen), it uses a domain-invariance
objective so it does not quietly memorize one machine's signature, and it defers
its least confident cases to a human instead of guessing. The headline number is
not in-distribution accuracy; it is how much of the cross-clinic accuracy drop
the model's own uncertainty lets it recover.

The backbone is swappable on purpose. The contribution is the cross-cohort plus
trustworthy-deferral system, not a particular network.

## What to clone

Clone this repo (your own copy) and build on it. Do not fork any single upstream
model. TB-Net, the obvious starting point, is TensorFlow and locked to one
backbone; DomainBed is a heavy benchmark harness built around ResNet. Both are
worth reading, neither is a good base for an edge-focused, backbone-agnostic
project. So this scaffold is fresh PyTorch, and it pulls the good parts of those
projects in as dependencies or as code you copy deliberately. The repos worth
cloning alongside this one, only to read and borrow from, are listed in
`docs/ONBOARDING.md`.

## Quickstart

### Local (with a GPU, for training)

    git clone <your-remote>/xctb.git
    cd xctb
    bash setup.sh          # makes a venv, installs deps, runs the smoke test

`setup.sh` installs the core dependencies and then runs
`scripts/smoke_test.py`, which checks the whole methodology (splits, deferral,
calibration) with no GPU and no images. If that prints "All smoke checks
passed," the environment is good.

Then get the data and build the manifest:

    # download the cohorts into data/ as described in docs/DATA.md, then:
    python scripts/build_manifest.py --data-root data
    python scripts/cohort_stats.py --manifest data/manifest.csv

And run the experiment:

    # in-distribution reference + every leave-one-cohort-out fold, baseline:
    python scripts/run_loco.py --config configs/base.yaml
    # same sweep with CORAL domain alignment, to compare:
    python scripts/run_loco.py --config configs/coral.yaml

`run_loco.py` writes a per-fold summary (gap, AURC, calibration error, and how
much deferral is needed to recover 90% of the gap) to `runs/loco_summary.csv`.
The difference between the baseline and CORAL summaries is the main result.

### Running on Kaggle

Kaggle already ships torch, torchvision, timm, numpy, pandas, scikit-learn and
pillow, so you do not need `setup.sh`. In a notebook:

    !git clone <your-remote>/xctb.git
    %cd xctb
    !pip install -q pyyaml torchmetrics pydicom
    !python scripts/smoke_test.py

Add the cohort datasets through "Add Data", point `--data-root` at the mounted
path, and run the scripts as above. The Montgomery and Shenzhen sets are small
enough to train comfortably on Kaggle's GPUs.

### No GPU, just checking the logic

    pip install numpy pandas scikit-learn pyyaml
    python scripts/smoke_test.py
    pytest -q

## Repo tour

    xctb/
      data/
        manifest.py      build one table across cohorts; validate; class balance
        splits.py        leave-one-cohort-out + random split, with leakage guards
        dataset.py       torch Dataset over the manifest (grayscale, DICOM-aware)
        transforms.py    image preprocessing, with a per-cohort override hook
        degradation.py   synthetic smartphone-capture degradation, continuous severity
        cohort_stats.py  quantify the domain shift (brightness/contrast/res)
      models/
        backbones.py     timm feature extractors (mobilenet, efficientnet, ...)
        model.py         backbone + classifier + optional domain head + FiLM
        grl.py           gradient reversal for DANN
        cohort_norm.py   cohort-conditional FiLM (the cross-field extension)
      losses/dg.py       Deep CORAL and IRM
      engine/
        train.py         one loop for ERM / CORAL / DANN / IRM
        infer.py         MC-dropout and deep-ensemble prediction + logits
      eval/
        metrics.py       AUROC, sensitivity, specificity, accuracy
        deferral.py      risk-coverage, AURC, and the gap-recovery metric
        degradation_uncertainty.py  does uncertainty actually rise with degradation?
      calibration.py     temperature scaling + ECE (numpy, no scipy)
    scripts/             build_manifest, build_degraded_eval, cohort_stats, train,
                          run_loco, evaluate, smoke_test
    configs/             base.yaml + coral / dann / cohort_film variants
    tests/               fast tests for the splits, degradation, and deferral math
    docs/                DATA.md (sources, licences, the confounding trap),
                          DEGRADATION.md (pipeline design, open ablation), ONBOARDING.md

The four files under `eval/` and `data/splits.py` are torch-free by design.
That keeps the tests fast and makes the split-and-deferral methodology reusable
on its own.

## The novel bit, in one paragraph

For a held-out cohort, take the accuracy drop relative to the same-cohort
(random-split) number. Then defer the model's most uncertain cases and ask how
much of that drop it recovers on the cases it keeps. A model whose uncertainty
is honest closes most of the gap by deferring a little; a model that is
confidently wrong on the new machine closes almost none, and that is precisely
what should stop it from being deployed. `xctb.eval.deferral` computes this,
and `scripts/smoke_test.py` shows it on synthetic data (oracle uncertainty
recovers the whole gap; random uncertainty recovers nothing).

## Status

Working and tested: the manifest, the splits with leakage guards, the deferral
and gap-recovery metrics, temperature-scaling calibration, the synthetic
smartphone-degradation pipeline, and the full torch-free pipeline. The training
loop, backbones, DG losses, and uncertainty inference are written and
syntax-checked but want a real GPU run to shake out. The cross-field extension
in `cohort_norm.py` is a working stub with the genuinely open problem marked.
See `docs/ONBOARDING.md` for who should pick up what.

Not done yet: the four cohorts are not downloaded (Montgomery/Shenzhen are a
direct download, NIAID needs a data-use agreement, RSNA needs a Kaggle
account — see `docs/DATA.md`), so nothing in this repo has run against a real
image yet, only `synthetic_manifest()` and synthetic test images. The `gan`
and `rephoto` degradation-comparison strategies in `docs/DEGRADATION.md` are
unimplemented placeholders, and the degradation-vs-uncertainty correlation
check has only been exercised against fabricated uncertainty, not a trained
model.

## Licence

Not set yet. Pick one before release (workshops usually expect MIT or Apache-2.0)
and add a LICENSE file. Note the NLM data terms in `docs/DATA.md`.
