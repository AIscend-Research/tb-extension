# Degradation pipeline: design and open questions

Phase 2 asked for a synthetic smartphone-degradation pipeline plus a way to
tag every image with clinic, TB-status, and degradation type/severity, and to
generate weak-supervision labels for an uncertainty head. This is what got
built, what changed from the original plan, and why.

## The pipeline

`xctb/data/degradation.py` implements eight continuous-severity degradations
(`severity` in `[0, 1]`, `0` is always a no-op):

- `defocus_blur`, `motion_blur` — loss of sharpness, two different causes
- `glare` — a bright Gaussian blob, like a lightbox reflection
- `shadow` — a smooth directional darkening, non-uniform lighting
- `rotation` — a handheld off-angle shot
- `compression` — JPEG re-encoding artifacts
- `resolution` — downsample/upsample, a low-resolution capture of a large film
- `noise` — sensor noise

It is PIL + numpy only (no scipy/cv2), matching the rest of `xctb/data`.
`compose_degradation(img, severity, strategy=...)` applies a named strategy's
kinds at that overall severity, jittering each kind's actual severity around
the target so a composed image is not identically scaled in every channel. It
returns the degraded image and a dict of exactly which per-kind severities got
used, so nothing is a black box.

Tagging is `build_degradation_manifest`: it expands a manifest into one row
per (image, severity), keeping the existing `cohort` and `label` columns and
adding `degradation_strategy`, `degradation_severity`, `degradation_seed`.
Pixels are not touched and no files are copied — `xctb.data.dataset.CXRDataset`
applies `compose_degradation` on the fly at load time, keyed by those three
columns, so the same row degrades identically every epoch. This matters
because there are no real cohort images checked into this repo yet (data
sourcing is still a manual, account-gated step, see `docs/DATA.md`); an
on-disk degraded copy of nothing is not useful, and once real images exist
this design also avoids doubling the dataset on disk.

## The strategy-comparison ablation

The roadmap's extension idea was to compare degradation-simulation strategies
(hand-specified blur+noise vs. a learned GAN phone-camera model vs. actual
re-photography of printed films) and treat the comparison itself as a
contribution. `DEGRADATION_STRATEGIES` in `degradation.py` names four:

- `simple` — implemented. Defocus blur + noise only, the baseline other
  degradation papers use.
- `full` — implemented. All eight kinds composed.
- `gan` — **not implemented**. Needs a phone-camera degradation model trained
  on paired clean/phone-photo data, which we do not have.
- `rephoto` — **not implemented**. Needs actual printed films re-photographed
  with a phone, to check whether synthetic severity tracks real severity.

Calling `compose_degradation(..., strategy="gan")` or `"rephoto"` raises
`NotImplementedError` naming what is missing, rather than silently falling
back to `full`. Treat closing that gap (sourcing or generating a small set of
real re-photographed images, even 20-30, and comparing `full`'s output against
them on the quality proxies already used in `xctb/data/cohort_stats.py`
— brightness, contrast, resolution) as the actual open item here, not the two
implemented strategies.

## Why there is no trained confidence head

The roadmap's weak-supervision idea was: heavy degradation gets a "high
uncertainty is appropriate" label, and a confidence head is trained on it.
That was written before the uncertainty-methods survey (Phase 1, Chris)
compared MC-dropout, deep ensembles, evidential deep learning, and temperature
scaling, and picked **MC-dropout + temperature scaling**. Both of those are
label-free at the image level — MC-dropout reads the spread of the model's own
dropout-perturbed predictions, and temperature scaling is fit on validation
logits, not on a degradation-derived target. Building a second, separate
confidence head trained on the weak labels would mean carrying two
uncertainty mechanisms with no clear way to reconcile them if they disagree,
so it was left out here.

The weak label was not dropped, though: `severity_to_target_uncertainty`
still produces it, and `xctb/eval/degradation_uncertainty.py` uses it as a
**validation** check instead of a training target — correlate it (Spearman)
against whatever uncertainty `xctb/engine/infer.py` actually produces for the
same images. A model whose MC-dropout uncertainty rises with degradation
severity is behaving the way the "retake photo" deferral message assumes; one
whose uncertainty doesn't move is a red flag worth catching before the
Phase 4 deferral results are trusted. If a later iteration does want a
supervised confidence head, `severity_to_target_uncertainty` is exactly the
label to train it on — nothing here forecloses that.

## What is still open

- No real images exist in this repo, so nothing above has been run against an
  actual smartphone re-photograph yet. Everything is exercised via
  `synthetic_manifest()` and a synthetic checkerboard image in `tests/`.
- The `gan` and `rephoto` strategies are unimplemented placeholders, not
  future-proofing cruft — pick one up if the ablation is worth the real
  contribution slot.
- `uncertainty_vs_severity` has not been run against a trained model yet
  (there is no trained model), only against fabricated
  honest/dishonest uncertainty arrays in `tests/test_degradation.py`. Run it
  for real once `scripts/run_loco.py` produces predictions on a
  `build_degradation_manifest` eval set.
