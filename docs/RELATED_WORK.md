# Related work: lightweight TB CXR screening papers

Phase 1 of `ONBOARDING.md`: skim TB-Net and a couple of lightweight-TB-CXR papers
to place our evaluation protocol against SOTA. Notes on TB-Net and LightTBNet
already existed in the team tracker; this adds three more, chosen to cover the
range of "lightweight" claims and to stress-test whether anyone already does
cross-cohort evaluation (they don't, with one partial exception below).

## TB-Net (Wong et al., 2022)

- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC9022489/
- Repo: https://github.com/darwinai/TuberculosisNet
- Self-attention CNN, architecture found via generative synthesis under
  constraints (sensitivity/specificity >= 95%, <= 5M params).
- Data: Montgomery + Shenzhen + NIAID + RSNA pooled, 6939 images after quality
  screening (3461 pos / 3478 neg).
- **Eval protocol: random 80/10/10 split over the pooled multinational data.**
  No cross-cohort or leave-one-cohort-out evaluation.
- Result: 99.86% accuracy, 100% sensitivity, 99.7% specificity. 4.24M params,
  0.42B MACs.

## LightTBNet (Capellán-Martín et al., 2023)

- Paper: https://arxiv.org/pdf/2309.02140
- Repo: https://github.com/dani-capellan/LightTBNet
- Deep CNN, CLAHE preprocessing, 256x256 input.
- Data: Montgomery (138 CXR) + Shenzhen (662 CXR) only.
- **Eval protocol: 20% held out for test, remaining 80% split via 5-fold CV**
  for hyperparameter tuning. Folds are drawn from the same pooled MC+SZ data,
  not held out by cohort.
- No cross-cohort or LOCO evaluation.

## Pasa et al., "Efficient Deep Network Architectures for Fast Chest X-Ray
Tuberculosis Screening and Visualization" (Scientific Reports, 2019)

- Paper: https://www.nature.com/articles/s41598-019-42557-4 (open access via
  PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC6472370/)
- The most genuinely lightweight model of the group: a 5-block CNN (two 3x3
  convs + maxpool per block, 1x1-conv shortcuts, batchnorm, GAP + softmax),
  **~230K parameters** — roughly 18x smaller than TB-Net and small enough to be
  the strongest "edge-deployable" baseline to cite.
- Data: Montgomery (138), Shenzhen (662), Belarus (304 TB-confirmed) — three
  cohorts, evaluated **separately**, plus a pooled "combined" run.
- **Eval protocol: 5-fold CV run independently within each dataset** (and again
  on the pooled combined set). This is the closest any of these papers gets to
  acknowledging cross-site variation — they report per-cohort numbers side by
  side — but folds never cross cohort boundaries, so it is not LOCO.
- Results are notably lower than TB-Net/LightTBNet's pooled numbers and vary a
  lot by cohort: Montgomery 79.0% acc / 0.811 AUC, Shenzhen 84.4% acc / 0.900
  AUC, combined 86.2% acc / 0.925 AUC. Worth citing as evidence that per-cohort
  difficulty already varies substantially even before you test cross-cohort.

## Ensemble CNN framework, Tianjin Haihe Hospital + external validation
(Scientific Reports family, 2024, PMC11301748)

- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11301748/
- 43-model ensemble grid over 13 backbones (AlexNet, DenseNet, EfficientNet,
  GoogleNet, MobileNet, VGGNet, plus two TB-specific nets) x 3 fusion strategies
  (voting, attention, concatenation). Not lightweight by itself, but relevant
  because of its eval protocol.
- Data: primarily a private set from Tianjin Haihe Hospital (2191 images), with
  **Shenzhen and Montgomery used purely as external validation sets** — i.e.
  train on Haihe, test on SZ/MC without retraining.
- **This is the one paper in the set that does something LOCO-adjacent**: it
  explicitly reports a generalization drop, describing "dataset dependency"
  and "low scores when validations are performed across the dataset."
  Important precedent to cite and immediately distinguish from our work:
  - it's a single train->external-test direction (Haihe -> SZ/MC), not a full
    leave-one-cohort-out sweep over all cohorts;
  - it reports the drop but does nothing about it — no calibrated uncertainty,
    no deferral, no measurement of how much of the drop is recoverable.
  - That's exactly the gap our delta statement claims.

## Lightweight hybrid GhostNet + MobileViT (2025, PMC12731716)

- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12731716/
- Hybrid CNN (GhostNet, 3.9M params) + transformer (MobileViT-XS, 1.94M
  params) branches, pooled and concatenated. 7.73M params total, 282M FLOPs.
- Data: a 7000-image CXR dataset (3500/3500) and TBX11K (1600 images),
  evaluated **separately**, not pooled and not cross-tested against each other.
- **Eval protocol: random 70/15/15 split + 5-fold CV**, both within-dataset.
  No cross-dataset/cross-cohort evaluation between the two datasets it uses,
  despite having two datasets available to do exactly that.
- Results: 99.5% / 99.2% accuracy on the two datasets respectively — again,
  in-distribution numbers only.

## Net takeaway for the intro / related-work section

Every lightweight-TB-CXR paper surveyed (TB-Net, LightTBNet, Pasa et al.,
GhostNet+MobileViT) reports only in-distribution accuracy via random split or
within-cohort k-fold CV. The one paper that does cross-site testing (the Haihe
ensemble study) confirms a real generalization drop but treats it as a
limitation to note, not a metric to close — no uncertainty quantification, no
deferral, no systematic LOCO across cohorts. This backs the delta statement in
`README.md` ("The novel bit, in one paragraph") and closes out Chloe's earlier
"has anyone done LOCO on these datasets" check with a concrete citation to
point to when reviewers ask the same question.
