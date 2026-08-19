# Real re-photography

Nothing in this directory is committed except this file and the manifest template
(same reasoning as `data/raw/`: derived imagery with its own licensing terms, see
`docs/phase1_framing.md` section 4).

There are two separate questions people come here for, they need different data,
and conflating them is the main way this directory gets misused.

| question | what it needs | tool |
|---|---|---|
| **Does the estimator work on a real phone?** Is the recovered optical density right, is the measured blur right, does the certificate's detectability claim hold? | a **printed phantom** whose density is known, photographed by real phones | `scripts/make_phantom_film.py` → `scripts/validate_real_recapture.py` |
| **Is our simulated degradation realistic?** Does `data/degradation.py` land in the same feature distribution as real phone photos? | real photos of **chest films**, any films, no ground truth needed | `scripts/ablate_degradation.py` |

Part 1 is the higher-leverage one and it is what the rest of this file is about.
Part 2 is unchanged and lives at the bottom.

---

## Part 1: the phantom pilot

### Why a phantom and not a real film

Everything in the physics track is currently validated against `physics/film.py` —
a forward model written in this repo. `scripts/validate_physics.py` shows the
estimator recovers the parameters that model was handed, which is a necessary
check and a closed loop. It cannot fail for the interesting reason.

Opening the loop needs a real phone on one side and a **known density** on the
other. Neither of the obvious options gives both:

* **Real clinical films** have the fiducials and the right physics, but no
  ground-truth density map. A pilot on real films can measure repeatability and
  whether an effect moves in the right direction, not whether a number is right.
* **Printed archive PNGs** have neither. Public archives are cropped, so they
  carry none of the three fiducials the inversion needs — which is exactly what
  `scripts/audit_fiducials.py` measured at 5.5% coverage on NLM.

So the sheet is manufactured. The phone, the lens, the veiling glare, the
lightbox, the sensor and the JPEG encoder are all real; only the film is made to
order, and made to order is the point — it is what gives per-region truth.

### What you need

| item | cost | why |
|---|---|---|
| Inkjet or laser **transparency film**, 2 sheets | ~£1 | the whole rig is a transmission measurement; paper is useless |
| **Matte black tape** or black card | ~£2 | the printed rim is nowhere near opaque enough to act as the beam stop |
| A **calibrated step wedge** (Stouffer T2115 or equivalent) | ~£20 | the only absolutely calibrated object in the rig; every density reported on real captures is referred to it |
| A **lightbox**, or a white LED panel, or a tablet showing pure white | — | the illuminant |
| **2+ phones** | — | artifact diversity a single hand-tuned model cannot produce |

The wedge is the one thing not to skip. Without it the tone curve can only be
checked for relative agreement, and "the estimator's densities are self-consistent"
is not the claim anyone is asking about.

### Steps

```bash
# 1. Generate the sheet (A4 portrait by default)
python scripts/make_phantom_film.py --out outputs/phantom --print-long-mm 297

# 2. Print and assemble it -- read outputs/phantom/PRINTING.md, all of it.
#    Colour management OFF. Tape the rim. Seat the wedge in its lane.

# 3. Shoot: 5 reference frames (best conditions you can manage) and ~40 test
#    frames varying phone, angle, distance, room light and focus.

# 4. Log every frame in a copy of manifest_template.csv, then:
python scripts/validate_real_recapture.py --phantom outputs/phantom \
    --manifest data/real_recapture/manifest.csv --out outputs/real_recapture
```

Try the analysis before you print anything:

```bash
python scripts/validate_real_recapture.py --phantom outputs/phantom --dry-run
```

That synthesises captures through `physics/film.py` and runs the identical
analysis. It proves the path works. It proves nothing about a phone — it is
`film.py` on both sides again, which is the situation this whole exercise exists
to escape.

### The manifest

`manifest_template.csv` — copy it, don't edit it in place.

| column | what it means |
|---|---|
| `capture_path` | path to the image file |
| `role` | `reference` (the print is characterised from these) or `test` (the experiment) |
| `phone_model` | free text, grouped on in the report |
| `condition` | short label for the manipulation, e.g. `angle15`, `room_light_on` |
| `angle_deg` | approximate off-axis angle |
| `distance_frac` | how much of the frame the sheet fills, 1.0 = filling it |
| `room_light` | `off` / `on` / `window` |
| `focus` | `locked` / `soft` |
| `notes` | anything you noticed while shooting; the glare cases especially |

A frame with no row cannot be used: the condition columns are what the analysis
groups by, and "some photos" is not an experiment.

### What comes out

`docs/REAL_RECAPTURE.md` states the four pre-registered gates and what each one
would mean if it failed. Briefly: recovered density within 0.05 OD of the
wedge-referenced truth after removing a constant offset; the estimator's PSF
within a factor of two of the phantom's own slanted edge; no more than 5% of
discs cleared with margin by the certificate but invisible to a matched filter;
and reference captures that agree with each other to 0.03 OD before any of that
is believed.

### Honest limits of this rig

* A printed transparency is not silver halide. It has no grain, a maximum density
  around 1.6 rather than 3.2, and a different surface — so specular glare behaves
  differently from a real film in a real folder. Findings about *the estimator's
  maths* transfer; findings about *how bad real archive photographs are* do not.
* The rim is tape, so the beam stop is better than a real direct-exposure region
  rather than worse. Veil estimates from this rig are a best case.
* Everything is referred to the wedge, so the wedge's own calibration is the
  floor on every claim. Stouffer quotes ±0.02 OD; nothing here can be tighter.
* One sheet, one printer. Print scale, ink and film stock are all uncontrolled
  variables until someone repeats it elsewhere.

---

## Part 2: degradation realism (unchanged)

`data/degradation.py` (physics-based) and `data/degradation_learned.py` (learned
generator) are both stand-ins for a clinician photographing a film.
`scripts/ablate_degradation.py` scores each stand-in against real captures — and
it needs real captures to score against. It does **not** need the phantom: any
real phone photos of chest films will do, ground truth or not.

**Option A: collect a small set.** ~20–30 public Montgomery/Shenzhen images,
printed or shown full-screen on a second monitor, 3–6 phone photos each varying
angle, framing, ambient light and device. Save as
`data/real_recapture/<clinic>/<orig_stem>__<phone_model>__<condition>.jpg`,
keeping the original stem so the pair is recoverable for a *paired* ablation.

```bash
python scripts/ablate_degradation.py --real-dir data/real_recapture/montgomery \
    --source data/raw/montgomery --out outputs/degradation_ablation.json
```

**Option B: use an existing set.** PhysioNet `cxr-phone` and Stanford CheXphoto
(the real-photo subset) are existing smartphone-photo-of-CXR collections. Neither
is TB-labelled or from this project's clinics, so neither can replace the LOCO
evaluation data — but either works as the "real" reference distribution for the
ablation. Check access terms first; PhysioNet may require a signed DUA.

**What "done" looks like here:** an ablation report with the `real` group
populated, i.e. a `*_vs_real` entry in the `separability` section of its JSON, and
the interpretation — whichever of `physics`/`learned` has the lower classifier
accuracy against `real` is the better stand-in, and that is what the training and
eval pipeline should default to.
