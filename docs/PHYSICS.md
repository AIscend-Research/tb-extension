# Physics-derived uncertainty: measuring the capture channel instead of learning it

Every chest radiograph carries its own calibration targets. Measure them and you can invert the
phone capture and bound what the photograph could possibly have carried — no labels, no
network, no training data.

This document is the reference for `src/tbtrust/physics/`. Read
[`density.py`](../src/tbtrust/physics/density.py) first: the sign convention is inverted from
the project brief and everything else depends on getting it right.

---

## 1. The three free calibration targets

| object on the film | known optical density | what it measures | module |
|---|---|---|---|
| lead L/R side marker | `D_min` ≈ 0.20 (base+fog) | bright densitometry anchor; the only *interior* sample of the lightbox | `tone.py` |
| direct-exposure region | `D_max` ≈ 3.20 | the optical **beam stop** → veiling glare | `glare.py` |
| collimation border | a hard `D_min`→`D_max` step | the capture **PSF**, by ISO 12233 slanted edge | `psf.py` |
| film outside the border | `D_min` | full-frame flat field → illumination surface | `tone.py` |

### The sign convention, which the brief has backwards

Two opposite conventions are in play and they must not be mixed.

- **X-ray transmission.** Lead transmits nothing, so in beam-stop terms the marker is "zero
  signal". This is the convention the project brief uses.
- **Developed film.** Film is a negative: more exposure → more silver → higher optical density
  → darker on a lightbox. Lead *blocks* the beam, so the film beneath it is barely exposed, so
  it sits at base+fog, so it is **transparent — the brightest thing in the frame**. The
  direct-exposure region took the full unattenuated beam, developed to maximum density, and is
  **the darkest**.

Everything downstream of the film — which is everything here, since we only ever see
photographs of developed film — uses the second convention. So the roles swap relative to the
brief:

```
optical beam stop / coronagraph  =  direct-exposure region   (D_max, near-opaque)
bright densitometry anchor       =  lead marker              (D_min, near-clear)
```

The idea is unaffected. There are still two anchors of known density and one of them is
optically black; it is simply the other one.

---

## 2. The pipeline

```
photo (8-bit JPEG)
  └─ fiducials.detect        marker, collimation quad, beam stop  →  Coverage grade
  └─ invert.invert           fixed-point loop, 2 iterations:
        tone (bright anchors, black point pinned)
        → linearise → psf.estimate_psf     (slanted edge)
        → tone.fit_illumination            (D_min regions)
        → glare.estimate_veil              (beam stop)
        → re-fit tone with corrected anchors
     ⇒ CalibratedFilm: density map + a split error budget
  └─ floor.density_floor     per-pixel minimum resolvable ΔD, per finding
  └─ certificate.certify     DETECTABLE / MARGINAL / INSUFFICIENT / ABSTAIN
  └─ triage.triage           REPORT / RETAKE (+ instruction) / REFER
  └─ channel.channel_report  capacity in bits, and where the lost bits went
```

### The bound

For a finding with unit-amplitude profile `t(x)` and density contrast `ΔD`, a matched filter —
optimal for a known signal in additive noise, so nothing can beat it — achieves

```
SNR = ΔD · √E / σ_D ,      E = Σ_f |T(f)|² · MTF(f)²
```

and requiring `SNR ≥ k` (Rose criterion, `k = 5`) gives

```
ΔD_floor(x) = k · σ_D(x) / √E
```

`E` is the template energy surviving the blur. It handles the area gain of a large finding and
the contrast loss of a small one in one quantity, with no need to pick a "characteristic
frequency".

Written out, the per-pixel differential density noise is

```
σ_D = (γ / ln10) · σ_v / (v − c₀) · (1 + V/I)
```

Three things to notice. The tone exponent scales it, which is why γ is worth measuring. It goes
inversely with the *signal above the black point*, so a crushed region carries almost no density
information however many bits you spend. And the veil enters as `1 + V/I`, exactly the
reciprocal of the contrast compression it imposes — a 20% veil costs 20% of your density
resolution wherever it lands. That term usually dominates in a real clinic photo, and nobody
measures it, because measuring it needs a beam stop.

### Random versus systematic, and why the split is load-bearing

`invert.py` reports two error budgets:

- **`sigma_random`** — pixel noise, quantisation, the scatter of the veil surface fit. Differs
  between two nearby pixels.
- **`sigma_systematic`** — the γ prior, the gain, the illumination interpolation, film D_min /
  D_max tolerances. Shared across the frame.

**Only the random terms enter the floor.** The floor bounds a *difference* of densities measured
millimetres apart, and a common scale or offset error cancels in a difference. Merging them
would inflate the floor several-fold and turn a usable bound into a useless one. This is the
same reason the γ prior is tolerable at all: absolute density is uncertain at the few-percent
level, density *differences* are not, and differences are what radiology reads.

`floor.py` makes one further distinction. Veil-fit error is spatially **correlated** on the
scale of the glare field, so it does not average down across a large lesion the way white noise
does. Crediting it with the matched filter's area gain made the bound several times too
optimistic for consolidations in the detectability test — the worst possible failure direction —
so `FloorSpec.correlated_terms` excludes it from that gain.

---

## 3. What has been measured

From `python scripts/validate_physics.py` on the synthetic film (see §5 for how to reproduce):

| quantity | result |
|---|---|
| collimation corners | ~0.15 px error on a clean capture, ~2.7 px at severity 1.0 |
| PSF σ from the slanted edge | within ~15% of truth up to severity 0.75 |
| veil at the beam stop | recovered at 0.8–1.5× truth across the sweep |
| lead marker | found reliably on clean captures; often lost above moderate severity |
| **detectability calibration** | **median predicted/empirical ≈ 1.7** |

The last row is the one that matters. `ratio > 1` means the bound is **conservative**: it
declares information lost slightly before an optimal detector actually loses it. That is the
safe direction for a safety valve. Quote this number with any result — a bound reported without
its measured calibration is an assertion.

### The falsification test

`validate.detectability_experiment` inserts lesions of known contrast, pushes them through the
simulated capture, and lets a matched filter that already knows the lesion's position, size and
shape try to tell present from absent. It then compares the empirical d′ threshold against the
blindly-computed floor.

This is not circular, and the objection is worth answering directly. The floor is computed from
quantities the estimator had to *measure blind* — σ_D from a noise model fitted to the image, E
from an MTF read off the collimation border, the veil amplification from the beam stop. The
empirical threshold comes from the detector's behaviour on actual noisy captures. Nothing forces
them to agree: under-measure the veil and the floor comes out optimistic; over-smooth the MTF
and it comes out pessimistic. The ratio is a score on the estimator, and it is free to be wrong.

---

## 4. Known limitations, in order of how much they should worry you

1. **The veil is measured at the periphery and interpolated across the middle.** The beam stop
   is an annulus. A specular reflection sitting in the centre of the film is caught only when it
   is bright enough to push a pixel above base+fog — `glare._add_impossible_brightness` exploits
   the fact that nothing on a developed sheet is clearer than base+fog, so any excess is
   unambiguously stray light. A *dimmer* central reflection is under-reported and the
   certificate is optimistic there. **This is the leading known bias.**
2. **The finding contrasts are nominal placeholders.** `findings.py` ships physically sensible
   values marked `source="NOMINAL"`, not numbers from a published table. Relative statements
   depend only on the floor and are sound; absolute verdicts inherit the table's uncertainty.
   Replace it with `--findings table.yaml` before publishing.
3. **γ is a prior, not a measurement**, unless a third distinct density is present. See §6.
4. **The bound is about the measurement channel, not diagnostic difficulty.** The dominant
   obstacle to spotting a real nodule is anatomical clutter — ribs, vessels, the heart border —
   not photon noise. A lesion can clear this floor and still be invisible against a rib.
   `floor.anatomical_noise` estimates that clutter and `include_anatomical=True` folds it in,
   but the certificate deliberately does not: its claim is the narrow, falsifiable one, *this
   photograph destroyed information the film had*. Clutter is in the original film too, and no
   retake fixes it.
5. **`px_per_mm` is inferred** from the detected field against a standard cassette diagonal,
   ±20%. Pass a measured value when you can.
6. **The lead marker detector is the weakest component.** It is tuned to reject rather than
   guess (`MARKER_ACCEPT = 0.6`), because a false marker feeds the illumination fit a bogus
   interior sample. Losing it costs less than it sounds: the unexposed film outside the
   collimation border is the same `D_min` density and wraps the whole frame.
7. **All in silico.** Simulated degradation, no real phone recaptures. Same caveat as the rest
   of the project — see [`LIMITATIONS.md`](LIMITATIONS.md).

### Coverage is the load-bearing assumption

None of this works on an image whose fiducials were cropped away, and public archives crop.
`scripts/audit_fiducials.py` measures how often they survived, per clinic. **Run it first.** A
low certifiable rate is a result to report with its number, not a bug to work around — and the
simulated re-photography path in `film.py` remains fully available, since it paints the
fiducials back on and gives a controlled experiment with ground truth.

---

## 5. Running it

```bash
# 0. does the corpus even carry the fiducials?  (seconds; no model, no GPU)
python scripts/audit_fiducials.py --manifest data/processed/manifest.csv --out outputs/fiducial_audit

# 1. the experiments that could falsify the bound
python scripts/validate_physics.py --quick          # ~1 min, CI gate
python scripts/validate_physics.py                  # ~10 min, publication run

# 2. a certificate per image, joinable to the model's predictions by `path`
python scripts/physics_certificates.py --manifest data/processed/manifest.csv \
    --out outputs/certificates.csv --severities 0,0.25,0.5,0.75,1.0
```

**Working resolution matters more than any other flag.** The floor depends on how many pixels a
finding spans. A phone photographing a 35 cm film at 3000 px gets ~8 px/mm; at 320 px it gets
0.8, so a 2 mm miliary nodule is sub-pixel and the certificate correctly but uselessly calls
every image insufficient. 1024 is the smallest size at which the severity sweep separates
properly (miliary marginal on a clean capture, insufficient once degraded, larger findings still
carried), and it costs about 2 s per image.

Notebooks [`05`](../notebooks/05_fiducial_coverage_audit.ipynb) through
[`08`](../notebooks/08_physics_deferral_and_triage.ipynb) walk the whole track on Kaggle.

---

## 6. The one hardware ask

`γ` is the weakest assumption in the pipeline: with two anchors the fit pins the black point and
takes γ from an sRGB prior, propagating its uncertainty into absolute density. A **third
distinct density** breaks the degeneracy and `fit_tone` switches to fitting γ properly, reading
its uncertainty off the χ² curvature.

The cheapest way to get one in the field is to **tape a small step wedge to the lightbox beside
the film** — a few pence of exposed, processed film with two or three known densities. It turns
the weakest assumption in this pipeline into a measurement. A **ruler in the frame** likewise
settles `px_per_mm` exactly. Both are in
[`DEPLOYMENT_CHECKLIST.md`](DEPLOYMENT_CHECKLIST.md).

---

## 7. Why this is worth having alongside the learned uncertainty

Three things it buys that a confidence score cannot.

1. **Aleatoric uncertainty from first principles.** Immune to the standard critique — "you have
   just trained a blur detector" — because nothing is trained and the output has units of
   optical density. It can be checked, and it can be wrong.
2. **A principled retake/refer split.** "Defer" collapses two situations a clinic must
   distinguish: *the photo is bad* (thirty seconds and another shot — and the glare field says
   which way to move the phone, the PSF anisotropy says whether it was shake or defocus) versus
   *the photo is fine and the case is hard* (retaking reproduces the same image; refer). Only
   the physics separates them, because the floor is a property of the capture and the residual
   uncertainty is a property of the case.
3. **A Shannon framing that is arithmetic rather than analogy.** Phase 1 reached for channel
   capacity and could only gesture at it. `channel.py` computes bits and attributes the lost
   ones to the veil, the blur and the quantiser separately.

`eval/physics_deferral.py` puts the certificate margin on the same axis as the learned signals
so they can be compared head to head, and `complementarity()` measures whether the orthogonality
claim actually holds on your data. Errors the physics catches that nothing else does are the
confident-and-wrong cases — the ones a screening system most needs to catch.
