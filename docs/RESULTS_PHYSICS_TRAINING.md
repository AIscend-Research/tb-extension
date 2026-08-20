# Physics during training: measured, and it does not help

`docs/PHYSICS_IN_TRAINING.md` sets out the question — does measured capture
quality help the classifier **learn**, or only help it **abstain**? — and the six
arms that answer it. This is the answer, and it is negative.

**No arm improved anything. The one arm carrying real per-pixel physics was
significantly worse than the identical channel carrying a random floor map.**
The certificate's place is the post-hoc gate it already occupies.

## Setup

Montgomery held out, resnet18, 8 epochs, 3 seeds per arm, 18 runs. Every arm
trains on the same precomputed `film.simulate` captures at the same five
severities, so the only difference between arms is what the network may do with
the floor. Test is the held-out clinic at all five severities pooled: 138 images
× 5 = 690 rows. Differences are paired over seeds with a bootstrap interval,
because run-to-run variance on a corpus this size is larger than any effect
worth reporting.

Absolute numbers are **not** comparable to `configs/loco_*.yaml`: those train on
the on-the-fly degradation pipeline with continuous severity, these on a
five-point cached grid of a different capture model. The contrast between arms
is what this measures.

## Per arm

| Arm | test AUC | test accuracy | sensitivity | specificity |
| --- | --- | --- | --- | --- |
| control | 0.693 ± 0.024 | 0.664 ± 0.019 | 0.559 | 0.740 |
| **channel** | **0.653 ± 0.031** | 0.627 ± 0.008 | 0.324 | 0.847 |
| scramble | 0.694 ± 0.020 | 0.636 ± 0.038 | 0.223 | 0.936 |
| severity | 0.700 ± 0.038 | 0.637 ± 0.042 | 0.452 | 0.772 |
| weight_dn | 0.694 ± 0.049 | 0.611 ± 0.064 | 0.659 | 0.577 |
| weight_up | 0.713 ± 0.029 | 0.667 ± 0.019 | 0.410 | 0.853 |

### Paired against `control`

| Metric | channel | scramble | severity | weight_dn | weight_up |
| --- | --- | --- | --- | --- | --- |
| AUC | **−0.040** [−0.083, −0.016] | +0.001 | +0.008 | +0.002 | +0.021 |
| accuracy | **−0.037** [−0.054, −0.015] | −0.028 | −0.027 | **−0.053** [−0.104, −0.015] | +0.003 |
| sensitivity | **−0.235** [−0.348, −0.055] | **−0.336** [−0.479, −0.248] | −0.107 | +0.100 | −0.148 |

Bold = the interval excludes zero.

### Paired against `scramble` — the contrast the design exists for

`channel` against `control` only says that a fourth input channel changed
something. `channel` against `scramble` holds the extra channel fixed and
destroys only its *pairing to the image*, so it is the only comparison that is a
claim about the certificate.

| Metric | channel vs scramble | severity vs scramble |
| --- | --- | --- |
| AUC | **−0.041** [−0.066, −0.006] | +0.006 |
| accuracy | −0.009 | +0.001 |
| sensitivity | +0.101 | **+0.229** [+0.076, +0.410] |

**The real floor map is significantly worse than a random one.** Not merely
uninformative — actively harmful, by about the same margin it is worse than
having no channel at all.

## Reading it

**A fourth channel is free; pairing it correctly is what costs.** `scramble`
(0.694) and `severity` (0.700) both match `control` (0.693) on AUC. The extra
input width, the extra stem parameters and the smooth low-frequency map all cost
nothing in ranking quality. Only `channel` (0.653) loses AUC. That is the whole
result, and it is exactly the comparison the two control arms were built to
make: without them, `channel`'s drop could have been blamed on the architecture
change rather than on the physics.

**The damage is uniform in severity, not concentrated where the physics is
loudest.** The obvious hypothesis for a harmful floor channel — the network
learns "this photograph is degraded, predict the majority class" — predicts the
loss should concentrate at high severity. It does not. `channel` is below
`control` at every point, including severity 0.0 (0.699 vs 0.742), where the
capture is clean and the floor map is nearly flat. Whatever the channel is
doing, it is not a degradation shortcut.

| severity | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
| --- | --- | --- | --- | --- | --- |
| control AUC | 0.742 | 0.715 | 0.701 | 0.660 | 0.662 |
| channel AUC | 0.699 | 0.661 | 0.669 | 0.623 | 0.613 |

**Most of the accuracy and sensitivity movement is the operating point, not the
model.** Every four-channel arm trades sensitivity for specificity at the fixed
0.5 threshold — `scramble` most dramatically, at 0.223 sensitivity against
0.936 specificity — while its AUC is unchanged. Ranking is intact; the threshold
has moved. Accuracy and sensitivity deltas here should not be read as capability
differences, which is why AUC leads. The same applies to `weight_dn`'s accuracy
drop (−0.053) alongside an unchanged AUC (+0.002): it shifts the operating point
towards sensitivity (0.659 vs 0.559) and pays for it in specificity.

**Neither loss-weighting direction did anything.** `weight_dn` was the
argued-for one — down-weight photographs whose certificate says the finding
cannot be carried, on the grounds that their gradient teaches a shortcut. Its
AUC delta is +0.002. `weight_up` is nominally the best arm at +0.021 AUC, but
its interval spans zero and it is the arm with no prior reason to work, so the
honest reading is "no effect" rather than "the inverted hypothesis won".

## What this does and does not license

It licenses: **leaving the certificate where it is.** The physics track earns
its place at deferral — `eval/physics_deferral.py` shows the certificate ranks
cases the learned signals miss — and that is a different mechanism from helping
the network learn. This experiment says the second does not follow from the
first.

It does not license "the physics is uninformative". Specifically not settled:

* **Power.** Three seeds on one fold with 138 held-out images. The design can
  see an AUC effect of roughly 0.04 and would miss 0.02. A real but modest gain
  is not excluded.
* **One fold.** Montgomery only. The cross-site claim this project cares about
  needs the rotation, and `docs/SOURCES.md` is about getting more of it.
* **One representation.** The channel is the *worst finding's* floor map,
  log-normalised on train-split percentiles. A per-finding stack, the
  `sigma_d` field before the Rose factor, or the limiting-factor map are all
  different inputs and untested.
* **The abstentions.** 19% of the cache has no certificate and receives a
  constant worst-value map — a large flat patch, on a fifth of the training set,
  which is itself a strange input. Whether that alone accounts for `channel`
  losing to `scramble` has not been separated out, and it is the first thing to
  test if anyone revisits this.
* **The capture model.** These are simulated captures of archive scans, and the
  arms inherit whatever the forward model gets wrong.

Raw per-run results in `docs/figures/run1/physics_in_training.csv`, paired
summaries in `physics_in_training_summary.csv`.
