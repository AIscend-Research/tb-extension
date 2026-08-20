# Physics during training, not only at deferral

The certificate is currently a post-hoc gate. `eval/physics_deferral.py` takes a
trained network's calibrated probabilities and re-ranks them by whether the
photograph could carry the finding at all; the network itself never sees the
measurement. That leaves the obvious question untested:

> Does measured capture quality help the classifier **learn**, or only help it
> **abstain**?

This is the rig that answers it — the per-pixel density floor fed in as a fourth
input channel, or as a per-sample loss weight — and, as importantly, the two
control arms that decide whether a positive answer means anything.

## 1. Two obstacles, and what they forced

**The training degradation and the physics capture model are different
pipelines, and only one of them can be certified.** `data/degradation.py` applies
blur, glare, shadow, angle, JPEG and downscale straight to an archive scan. That
image has no lead L/R marker, no direct-exposure region and no collimation
border, so `physics/invert.py` has nothing to calibrate against and the
certificate **abstains on every training image**. The physics arms therefore have
to train on `physics/film.simulate` captures, which lay the fiducials on the film
before photographing it.

That is not a detail of convenience. It means a comparison against a model
trained the ordinary way would confound two changes at once, so **every arm here
— including the no-physics control — trains on the same `film.simulate`
captures**. The contrast is clean; the absolute numbers are not comparable to
`configs/loco_*.yaml`, and are not reported as if they were.

**The floor costs ~0.2 s per photograph.** Inversion runs a tone fit, a
slanted-edge PSF estimate and a glare surface fit, then an FFT per finding —
longer than a forward-backward pass. So it is precomputed once per (image,
severity) into `data/processed/physics_cache` and cached. The cost is that
severity becomes a five-point grid instead of a continuous draw. Every arm
inherits the same grid, so the contrast pays nothing for it.

The capture seed is `utils.seed.capture_seed(path, severity, seed)` — the same
CRC32 the certificate corpus uses — so a photograph in the training cache is bit
for bit the one `scripts/physics_deferral_real.py` scored. The training cache and
the evaluation corpus are the same photographs, not two draws of the same
process.

## 2. The arms

| Arm | Fourth channel | Loss weight |
| --- | --- | --- |
| `control` | — | — |
| `channel` | normalised per-pixel floor | — |
| `scramble` | **another image's** floor map | — |
| `severity` | one constant: the applied severity | — |
| `weight_dn` | — | down-weight where the certificate says the photograph cannot carry the finding |
| `weight_up` | — | the same weighting, inverted |

**The controls are the design.** Without them a positive result is
uninterpretable:

* `scramble` has the identical marginal distribution to `channel` and none of the
  pairing. A smooth, low-frequency extra channel can act as a regulariser and buy
  accuracy while carrying nothing about the image it is attached to. If `channel`
  beats `control` and `scramble` beats it by the same margin, **the physics
  contributed nothing** and the finding is "a fourth channel helps".
* `severity` is a single scalar the simulator already knew. If it matches
  `channel`, the per-pixel measurement — the expensive, novel part — is not what
  is paying, and the honest write-up is "capture severity helps, and you do not
  need a certificate to get it".

`weight_dn` is the argued-for direction: when the certificate says the photograph
cannot carry the finding, the archive label is still TB or not-TB, but nothing in
*this image* supports it, so the gradient teaches the network to predict the label
from whatever spurious cue is left. That is the definition of a shortcut, and
down-weighting should reduce it. `weight_up` is the hard-example-mining intuition,
equally plausible a priori, and only measurement separates them. Weights are
floored at 0.25 rather than driven to zero: a large fraction of this corpus is
INSUFFICIENT for the worst finding, and a hard zero would confound "physics
helps" with "less data hurts".

## 3. Three things that would have made this measure nothing

Each is pinned by a test in `tests/test_physics_training.py`.

**The channel starts inert.** The stem convolution is widened from 3 to 4
channels with the new kernel **zero-initialised**, so at initialisation the logit
does not depend on the fourth channel at all — feed it noise, zeros or a constant
and the output is bit for bit identical. The physics arm therefore starts as the
same function as its control, and any divergence afterwards is something the
channel bought. The usual recipe — copying the mean of the pretrained RGB weights
into the new channel — would perturb every prediction from step zero and confound
"the physics helped" with "the stem was reinitialised".

**The normalisation is shared, and fitted on train only.** The floor spans orders
of magnitude, so the channel is `log10` scaled between percentiles fitted on the
training split. Per-image normalisation would have been the natural thing to
write and would have erased exactly the between-image differences the channel
exists to carry — a version of this experiment that runs, produces plausible
numbers, and measures nothing.

**An abstention is the worst value, not zero.** 27% of this corpus has no
certificate at all (`no_fiducials`). Handing the network a zero there would read
as "no degradation", the exact inverse. Abstentions get the clipped worst floor,
and in the weighting arms the floor weight.

## 4. Running it

```bash
# ~15 min on 8 cores for 800 images x 5 severities
python scripts/build_physics_cache.py --out data/processed/physics_cache

python scripts/measure_physics_in_training.py --seeds 3
```

Results are paired over seeds against the control, with a bootstrap interval on
the difference. Paired because the arms share seeds: on a corpus this size the
run-to-run variance is larger than any effect worth reporting, and the unpaired
difference would drown it.

## 5. Results

**Negative.** No arm improved anything, and `channel` -- the arm carrying the
real per-pixel floor -- was significantly *worse* than `scramble`, the identical
channel carrying another image's floor map (AUC -0.041, interval excluding
zero). A fourth channel is free: `scramble` and `severity` both match `control`
on AUC. Pairing it correctly is what costs. The certificate's place is the
post-hoc gate it already occupies.

Full numbers, the severity breakdown, and what the result does not license:
`docs/RESULTS_PHYSICS_TRAINING.md`.
