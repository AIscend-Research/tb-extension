# Onboarding: where the work is

This maps the project roadmap onto files in this repo so you can pick something
and start. Each item says which files to touch. The split and deferral code is
already written and tested; most of the open work is data wiring, the training
variants, and the cross-field extension.

Before anything: run `python scripts/smoke_test.py`. If it passes, your
environment is good and you understand the core objects (manifest, splits,
deferral). Read the two files it exercises, `xctb/data/splits.py` and
`xctb/eval/deferral.py`, because they define the experiment.

## Reference repos worth cloning (to read and borrow, not to fork)

    git clone https://github.com/darwinai/TuberculosisNet        # TB-Net baseline (TensorFlow)
    git clone https://github.com/facebookresearch/DomainBed      # DG algorithms, reference
    git clone https://github.com/jindongwang/transferlearning    # DeepDG: simpler DG toolkit (code/DeepDG)
    git clone https://github.com/torch-uncertainty/torch-uncertainty  # uncertainty + selective-classification metrics

We are not building on top of any of these (see the README for why). TB-Net is
TensorFlow and single-backbone; DomainBed is a heavy benchmark harness. Read
them for the algorithm details and copy ideas, not the scaffolding.

## Phase 1 — background and framing

- Confirm the novelty gap: has anyone done leave-one-cohort-out on these exact
  TB cohorts with calibrated deferral? Write the one-paragraph delta statement.
- Skim TB-Net, DomainBed's `algorithms.py`, and a couple of lightweight-TB-CXR
  papers. No code here yet; this feeds the intro and related-work sections.

## Phase 2 — data and preprocessing

- Download the cohorts and lay them out per `docs/DATA.md`.
- Run `scripts/build_manifest.py`; confirm the class-balance table and act on the
  single-class warning (this is where the confounding trap bites).
- Run `scripts/cohort_stats.py` to get the domain-shift numbers.
- Good first issue: if a cohort needs its own preprocessing, add it in
  `xctb/data/transforms.py::per_cohort_transforms` and justify it in DATA.md.
- Good first issue: build the balanced RSNA-normals + NIAID-positives cohorts if
  you decide to use those sources (loaders already exist in
  `xctb/data/manifest.py`).

## Phase 3 — model development

- Baseline (ERM): `configs/base.yaml` with `dg_method: none`. Run the random
  split for the in-distribution reference, then the loco sweep for the gap.
  Files: `xctb/models/model.py`, `xctb/engine/train.py`, `scripts/train.py`.
- Domain-invariance: CORAL and IRM losses are in `xctb/losses/dg.py`; DANN is the
  domain head plus gradient reversal in `xctb/models/grl.py` and the training
  loop. Run `configs/coral.yaml` / `configs/dann.yaml`. Good first issue: verify
  each objective actually lowers the generalization gap, and tune `dg_weight`.
- Uncertainty and deferral: MC-dropout and deep ensembles are wired in
  `xctb/engine/infer.py`; temperature scaling is in `xctb/calibration.py`. The
  deferral policy and metric are in `xctb/eval/deferral.py`. Good first issue:
  add predictive-entropy as an alternative uncertainty and compare its AURC to
  MC-dropout variance.
- Efficiency numbers: report params / FLOPs / latency per backbone. Good first
  issue: add a `scripts/profile.py` that prints `backbone.num_parameters()` and
  times a forward pass, so the "edge-deployable" claim is measured.

### The cross-field extension (the strongest novelty slot)

`xctb/models/cohort_norm.py` implements cohort-conditional FiLM modulation, the
channel-estimation-inspired idea from the roadmap. It works, but the important
question is unsolved: what do you condition on for a cohort the model has never
seen? The stub defaults to the mean training embedding. Better answers are the
actual contribution:

- fit a small adapter on a handful of unlabeled target images at test time, or
- average the k nearest cohort embeddings by image statistics, or
- predict the modulation from a few target images directly.

Run it with `configs/cohort_film.yaml` and ablate against `configs/coral.yaml`.
This is where a genuinely new result is most likely to come from.

## Phase 4 — evaluation

- `scripts/run_loco.py` already produces the per-fold summary (gap, AURC, ECE,
  deferral needed to recover 90% of the gap). Run it once per config and diff.
- Good first issue: a `scripts/plot.py` for risk-coverage and reliability
  diagrams (needs matplotlib from requirements-optional.txt).
- Good first issue: the cohort-pair generalization analysis (train on one, test
  on another, fill a matrix). Useful on its own, independent of our method.
- Optional: qualitative error analysis. Are false negatives concentrated in one
  cohort or one TB presentation?

## Phase 5 — writing and release

- The leave-one-cohort-out split scripts and the gap-recovery metric are the
  reusable contributions; keep them clean and documented so others can cite them.
- Fill limitations honestly: all-public retrospective data, small folds, and a
  domain shift limited to four cohorts rather than a truly novel clinic.

## How we work

- Keep the torch-free core (`splits`, `deferral`, `calibration`, `metrics`)
  torch-free. It is what makes the tests fast and the methodology portable.
- Add a test when you add a metric or a split rule. See `tests/`.
- Run `pytest -q` and `python scripts/smoke_test.py` before you push.
