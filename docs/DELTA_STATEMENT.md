# Delta statement

**One-liner** (from the team tracker): existing work reports in-distribution
accuracy; we report the accuracy drop under real cross-site deployment, and
whether calibrated deferral recovers it.

**Paper-ready paragraph** (also in `README.md` under "The novel bit, in one
paragraph" — kept in sync with that version):

> For a held-out cohort, we measure the accuracy drop relative to the same
> model's random-split, in-distribution accuracy on that cohort. We then defer
> the model's most uncertain cases and measure how much of that drop is
> recovered on the cases it keeps. A model whose uncertainty is honest closes
> most of the gap by deferring only a small fraction of cases; a model that is
> confidently wrong on the new site closes almost none — which is exactly the
> signal that should stop it from being deployed there un-reviewed.

**Why the gap is real** (evidence, see `docs/RELATED_WORK.md` for full notes):
TB-Net, LightTBNet, Pasa et al. (2019), and the GhostNet+MobileViT hybrid
(2025) all report only random-split or within-cohort k-fold accuracy — none
hold out an entire cohort. The one paper that does cross-site testing (Haihe
Hospital ensemble study, train on Haihe / test on Shenzhen+Montgomery) confirms
a real generalization drop ("dataset dependency," "low scores... across the
dataset") but stops there: no uncertainty quantification, no deferral, no
measurement of how much of the drop is recoverable. That's the specific gap
this project fills — Chloe's earlier check that "the LOCO gap exists" now has
concrete citations behind it.

**What "recovers it" means operationally**: `xctb.eval.deferral`'s gap-recovery
metric, computed per LOCO fold in `scripts/run_loco.py` (writes to
`runs/loco_summary.csv` — gap, AURC, ECE, and deferral-to-recover-90%-of-gap).
`scripts/smoke_test.py` demonstrates the two extremes on synthetic data: oracle
uncertainty recovers the whole gap, random uncertainty recovers none. The real
experiment is where our MC-dropout / temperature-scaling / ensemble numbers
(see `docs/UNCERTAINTY_SURVEY.md`) land between those two poles, per cohort and
per training objective (ERM vs. CORAL vs. DANN vs. IRM).

**Status**: statement drafted and now backed by citations (this doc). Not yet
run against real trained models — Phase 3/4 in `ONBOARDING.md` still need a
GPU run of `run_loco.py` across `configs/*.yaml` to fill in actual gap/AURC/ECE
numbers before this becomes a results section instead of a hypothesis.
