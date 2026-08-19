# The real-recapture pilot: what is being tested, and what would falsify it

**Status: pre-registered, no data yet.** Everything below was written before any
real capture existed. That is the only time thresholds can be set honestly, and
it is why this file exists separately from the results it will eventually carry.

## The problem this closes

Every number in the physics track is currently validated against
`src/tbtrust/physics/film.py` — a forward model written in this repository.
`scripts/validate_physics.py` measures whether `invert.py` recovers the
parameters `film.py` was handed. It is a real test, it has caught real bugs, and
it is a closed loop: the same set of assumptions is on both sides of the
comparison, so it cannot detect an assumption that is wrong about the world.

Concretely, the following are all currently unfalsifiable in this repo:

* that the ISP tone curve is a power law with an S-curve the estimator does not
  model — real ISPs do local tone mapping, and nothing here would notice;
* that veiling glare is a broad Gaussian halo — real lenses have flare structure,
  ghosting and a long non-Gaussian tail;
* that the sensor noise is shot plus read — real phones denoise aggressively,
  which correlates the noise and breaks the assumption the floor rests on;
* that a JPEG at quality 85 is the whole codec story — real pipelines are HEIC,
  multi-frame, and sharpened.

Each of those, if wrong in the direction that matters, makes the certificate
optimistic: it would clear photographs that cannot carry the finding.

## The instrument

`scripts/make_phantom_film.py` generates a printable phantom
(`src/tbtrust/physics/phantom.py`) carrying:

* the three fiducials `fiducials.detect` looks for, at the geometry
  `film.add_fiducials` paints them — so the blind estimator runs on it unmodified;
* an 11-step printed **density staircase**;
* a **detectability grid**: 30 discs, six density contrasts (0.010–0.320 OD) by
  five diameters (1.5–24 mm of film), bracketing three of the four findings in
  `findings.py` (consolidation is off the top of both ladders and is flagged as
  an extrapolation rather than a measurement);
* a 5° **slanted edge** inside the field, for an MTF measurement independent of
  the collimation border the estimator uses;
* a **millimetre scale**;
* a lane for a **calibrated transmission step wedge** (Stouffer T2115, 21 steps
  0.15 OD apart), taped inside the collimation field so it rectifies into a known
  place and needs no hand-marked corners on every frame.

The wedge is what makes the rig a measurement rather than a demonstration. A
printer's transfer is nonlinear and device-specific, so the densities
`phantom.build` targets are *not* what lands on the transparency. What lands is
measured, off the wedge, from the reference captures.

## The protocol

Two stages. The separation is the design, not bookkeeping.

**Stage 1 — characterise.** Five reference captures, best conditions available.
Rectify, read the wedge, fit a monotone pixel → OD transfer per capture, apply it
to every region. Average across captures. This is what the print carries.

**Stage 2 — score.** ~40 test captures across phone, angle, distance, room light
and focus. Each is handed to the blind estimator, which knows nothing about
wedges or phantoms. Its recovered density is compared against the Stage 1 truth.
The two paths share only the photograph.

`data/real_recapture/README.md` has the shopping list and the capture matrix.

## The four gates

Pre-registered thresholds, in `scripts/validate_real_recapture.py:GATES`. The run
exits non-zero if any fails.

### 1. Reference reproducibility ≤ 0.03 OD

The five reference captures must agree with each other before anything is
compared against them. This is the floor on every other claim: a rig whose own
reference wanders by 0.05 OD cannot detect a 0.03 OD estimator error.

*If it fails:* the capture setup is the problem, not the estimator. Usually
auto-exposure that was not locked, or the lightbox warming up.

### 2. Density RMSE after offset removal ≤ 0.05 OD

Recovered density against wedge-referenced truth, over the staircase and the grid
backgrounds, **after removing a constant offset**.

The split is the point. `invert.py` budgets its error in two parts — a systematic
term it admits it cannot pin (the gamma prior, the film's D_min/D_max tolerances)
and a random term. Only the random term enters the floor, because the floor bounds
a *difference* of densities millimetres apart and a common offset cancels. So a
large bias with small residual scatter is the pipeline behaving as documented; a
large residual after the offset is removed is the floor being wrong.

0.05 OD is roughly twice the lung-field `sigma_random` the estimator reports on
clean simulated captures, i.e. loose enough that ordinary print and wedge
tolerances do not trip it, tight enough that a factor-of-two error in the
recovered contrast does.

*If it fails:* the tone inversion does not survive a real ISP. That is the most
likely single failure of this whole pilot, and the most useful one, because it is
directly actionable — it would justify fitting the ISP curve from the wedge in
deployment rather than assuming a power law.

### 3. PSF agreement within a factor of two

The estimator's sigma (from the collimation border, in the photo frame, converted
by the rectification's linear scale) against the phantom's interior slanted edge.

Both are core-width measurements: the phantom edge is read from the 10–90% rise
rather than a second moment, precisely so that veiling glare — which puts a low,
wide halo under every edge — does not inflate it. The halo is `glare.py`'s job and
is measured separately off the beam stop.

*If it fails:* the MTF is being measured somewhere the findings are not. The two
edges differ in density range and in position, so a systematic gap between them is
a real statement about field dependence, not noise.

### 4. Clear detectability violations ≤ 5%

For every disc: `d'` from a matched filter on the recovered density, against the
certificate's predicted floor for a target of that size at that site.

A **clear violation** is a disc the certificate cleared with at least 3 dB of
margin and that the detector came nowhere near finding (`d' ≤ 0.7 k`). The raw
"cleared but invisible" count is reported alongside and deliberately *not* gated:
a disc a hair above the floor and a hair below `d' = k` is the bound being
approximately right at the hardest point, with noise on both sides of the
comparison. A gate that fires on those would fail on the forward model too, where
by construction there is nothing to catch.

*If it fails:* the certificate is optimistic on real hardware, and the density
floor needs a term it does not have. This is the result that would matter most —
it is the failure mode a screening deployment cannot tolerate, and it is the one
the whole physics track claims not to have.

## What a pass would and would not license

A clean run says: on this printer, this wedge, these phones and this lightbox, the
blind inversion recovers optical density to better than 0.05 OD once a constant
offset is removed, measures blur to within a factor of two of an independent edge,
and does not clear targets that a matched filter cannot find.

It would **not** say that the certificate is calibrated on clinical films. A
printed transparency has no grain, tops out near 1.6 OD instead of 3.2, and has a
taped rim that is a better beam stop than a real direct-exposure region — so this
rig is, if anything, generous. The next step after a pass is a handful of real
archive films photographed the same way, scored for repeatability and for
direction-of-effect only, since those carry no ground truth.

## Dry run

```bash
python scripts/make_phantom_film.py --out outputs/phantom
python scripts/validate_real_recapture.py --phantom outputs/phantom --dry-run
```

This synthesises captures through `film.py` and runs the identical analysis, which
is how the gates above were sanity-checked before anyone printed anything. On the
forward model at severities 0.15–0.65 it passes all four, with the raw
"cleared but invisible" rate around 8% and clear violations around 1%. Those are
closed-loop numbers and are not evidence about a phone; they are the answer to
"does this analysis discriminate at all, and are the thresholds in the right
place."
