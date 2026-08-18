# The interior-glare gap, measured

New-work item 3 asks whether the annulus geometry of the beam stop is the leading
optimism in the bound: the probes all sit around the rim of the collimated field,
the polynomial surface is then evaluated across an interior containing no probes,
and `docs/LIMITATIONS.md` states that "a dim central reflection is under-reported
and the certificate is optimistic there."

That has never been measured. `scripts/measure_interior_glare.py` measures it.

## Method

On synthetic captures the true veil field is known by construction
(`CaptureTruth.glare_field_true`), so the veil error can be split by region:

* **annulus** — inside `beamstop_mask`, where the probes actually sit. This is
  interpolation, and it is the estimator's best case.
* **interior** — inside `field_mask` but outside the beam stop dilated by a
  12 px guard. This is pure extrapolation.

20 synthetic chests x 5 severities = 100 captures, 80 of which detected a beam
stop and a usable interior. Size 384.

### One metric trap, worth recording

The first version of this script normalised the veil by the local `signal`, the
same way the floor consumes it. That is invalid for this comparison: **inside the
beam stop the signal is ~zero by construction** — that is what a beam stop is —
so `veil/signal` there reaches 4 to 41 and swamps any real regional difference.
It reported a median annulus "error" of 9.33 against 0.055 for the interior,
which is a statement about a vanishing denominator and nothing else.

Both regions are now divided by a single scalar: the median *true* signal over
the field interior. The numbers below are in those units.

## Result

`err = estimated - true`, so **positive means the estimator over-reports the
veil**.

| severity | annulus err | interior err | interior − annulus |
|---|---|---|---|
| 0.00 | +0.149 | **+0.778** | +0.556 |
| 0.25 | +0.128 | +0.314 | +0.056 |
| 0.50 | +0.070 | +0.192 | +0.053 |
| 0.75 | +0.043 | +0.216 | +0.077 |
| 1.00 | −0.002 | +0.275 | +0.166 |
| **overall** | **+0.070** | **+0.321** | |

Two findings, and the second is the one that matters.

**1. The interior really is the dominant error term.** Interior error exceeds
annulus error at every severity, by 4.6x overall (0.321 vs 0.070). The annulus,
where the probes are, is estimated well — its error falls to ~0 by severity 1.0.
So the regional hypothesis behind item 3 is correct: extrapolating a rim-only fit
across the field is where the veil error lives, and a second interior probe would
attack the right term.

**2. But the sign is the opposite of what the limitation claims.** The interior
veil is *over*-reported, not under-reported, at every severity tested. An
over-estimated veil means over-estimated noise, which means a floor that is too
**high** and a certificate that is too **strict**. In the interior the bound is
pessimistic, not optimistic.

`docs/LIMITATIONS.md` should be corrected: the interior is not where the
certificate quietly passes photographs it should fail. On this evidence it is
where the certificate fails photographs it should pass.

## Why this matters for the degenerate certificate

`docs/RESULTS_RUN1.md` §5 records that the certificate returns
`detectable = 0.00` at every severity **including zero**, pinned by miliary
nodule. The largest interior over-estimate here is at severity 0.00 (+0.778) —
i.e. on a pristine capture the estimator hallucinates interior veil where there
is essentially none, inflating the floor exactly in the condition that should be
easiest.

That is a plausible contributor to a clean synthetic capture being declared
insufficient, and it is testable: re-run `physics_certificates.py --severity 0`
with the interior veil forced to the annulus-fitted value and see whether
miliary's margin crosses zero.

## Caveats

* Synthetic captures only, against this repo's own forward model. The sign is
  consistent across 80 captures and all five severities, which makes it unlikely
  to be noise, but it is not evidence about a real phone.
* Region medians, not per-pixel error distributions. A compact specular hotspot
  could still be under-reported while the regional median is over-reported; that
  is the case `_add_impossible_brightness` exists to catch and it is not
  separately isolated here.
* The guard band is 12 px at size 384. It has not been swept.

## Reproduce

    python scripts/measure_interior_glare.py --n-images 20 --size 384 \
        --out outputs/interior_glare
