# More two-class clinics: what exists, and what it costs to add one

The rotation reports two honest holdout folds. `docs/LIMITATIONS.md` §3 is blunt
about why: the manifest draws on four sources, only Montgomery and Shenzhen
contain both classes, and a single-class holdout has an undefined sensitivity or
specificity plus a clinic label that is a near-perfect proxy for the diagnosis.
Two folds is a thin basis for a cross-site claim, and no modelling change
improves it. A third and fourth genuinely independent two-class site would.

This document is the survey, the audits a new source has to survive, and the
three things those audits already measured on the data in hand — one of which
closes off the route `LIMITATIONS` §3 currently recommends.

## 1. The candidates

`src/tbtrust/data/sources.py` holds this as a machine-readable registry;
`python scripts/audit_clinics.py sources` prints it. Ranked by what they buy
*this* project, which is not the same as ranked by size.

| Source | Country | n | Both classes | Resolution | Access | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| **NITRD DA** | India | 156 | yes | full | Kaggle, no gate | **usable** |
| **NITRD DB** | India | 150 | yes | full | Kaggle, no gate | **usable** |
| TBX11K | China | 11,200 | yes | 512², downscaled from ~3000² | Drive/Baidu link | usable with caveats |
| VinDr-CXR | Vietnam | 18,000 | yes | full DICOM | credentialed PhysioNet + CITI | usable with caveats |
| PadChest | Spain | 160,000 | yes | full | BIMCV registration | usable with caveats |
| Kaggle TB aggregate | mixed | 4,200 | yes | 512² | Kaggle, no gate | **not a fold** |
| NIAID TB Portals | multi | ~3,000 | no | varies | DUA | not a fold |

**Start with NITRD DA and DB.** They are the best value here by a distance:
roughly Montgomery-sized, genuinely two-class (~78 abnormal / ~75 normal each),
a third country and population, no access gate, and — unusually — DA and DB were
shot on *different X-ray machines at the same institute*. That last property is
worth as much as the extra fold: it is the within-site, between-machine shift
this project keeps assuming and has never been able to measure. Taken together
they take the rotation from two folds to four.

**TBX11K is the largest and CC BY 4.0, but it ships at 512×512**, downscaled
from ~3000×3000. It can serve the classifier and the cross-site claim; it cannot
serve the physics track, whose fiducial detection and PSF estimate need exactly
the resolution the downscale threw away. Its "sick-but-non-TB" class is also the
only thing here that could separate *abnormal* from *TB*, a harder and more
useful question than the one currently asked.

**VinDr-CXR is on the merits the strongest addition** — radiologist-labelled,
full-resolution DICOM, two named Vietnamese hospitals, TB as one of six global
labels — and the caveat is lead time, not quality. CITI training plus a
credentialed DUA is weeks. Start it before it is needed.

**PadChest** would test the cross-site claim harder than another Asian site, and
the resolution suits the physics track, but roughly three quarters of its labels
are text-mined from reports rather than read by a radiologist, and Alicante's TB
prevalence is low enough that the positive class will be small and possibly
dominated by sequelae rather than active disease.

**The Kaggle "Tuberculosis (TB) Chest X-ray Database" is not a clinic**, and it
is the most dangerous entry precisely because it is the most convenient — it is
what `scripts/download_data.py --kaggle-aggregated` fetches. It is a re-bundling
of NLM, Belarus, NIAID and RSNA. Adding it as a fifth clinic would put Shenzhen
images on both sides of a leave-one-clinic-out split while `data/splits.py`,
which keys on the `clinic` column and trusts it, reported a clean fold. Four
folds that quietly lie are worse than two that do not.

Fields the registry could not verify from a primary source are marked, and
`scripts/build_manifest.py` re-derives class balance from the files regardless.
Counts have a way of drifting between a paper's table and the tarball that
arrives.

## 2. The two audits a new source must survive

`scripts/audit_clinics.py`. Both answer questions the split code cannot.

### Overlap, on the pixels

A source that is secretly a re-host of one already present passes every
metadata-level check: new filenames, new folder layout, different size. So the
audit hashes the pixels — a difference hash, which survives the rescale and the
JPEG re-encode that a re-bundling applies.

**The threshold had to be calibrated, and calibrating it overturned the first
version of this check.** With the defaults ordinary for photo deduplication — an
8×8 grid and a threshold of 6 — the audit reported **half of Montgomery as
overlapping Shenzhen**, two sets that share no images. Chest radiographs are far
more alike than photographs, so the known-different distribution sits much
closer to zero than the usual advice assumes.

`audit_clinics.py calibrate` measures the two distributions it has to separate:
each image against its own simulated 512 px JPEG re-bundle (the duplicate the
audit must catch), and all pairs between two sets known to be disjoint.

| Hash grid | bits | duplicate ≤ | different ≥ | usable gap |
| --- | --- | --- | --- | --- |
| 8×8 | 64 | 2 | 4 | 2 bits |
| 12×12 | 144 | 2 | 9 | 7 bits |
| **16×16** | **256** | **2** | **26** | **24 bits** |
| 24×24 | 576 | 6 | 99 | 93 bits |

The defaults are now 16×16 at a threshold of 14 — the midpoint of the gap, taken
towards the permissive end deliberately, because a missed duplicate is a silent
leak and a false one costs somebody looking at two filenames.

Re-run at those settings, Montgomery against Shenzhen is **clean: zero pairs**.
Each clinic contains one within-clinic near-duplicate pair at 13–14 bits, right
at the threshold and probably borderline, flagged for eyes rather than acted on.

### Source confound: can capture statistics alone call the label?

`LIMITATIONS` §3 offers a way to manufacture two-class folds out of single-class
sources: mix one source's normals with another's positives. It works
arithmetically. The audit fits a **logistic regression on nine low-level capture
statistics** — brightness, contrast, dynamic range, histogram entropy, Laplacian
variance, pixel dimensions — and reports the cross-validated AUC. Nothing in that
feature set can see a cavity or an infiltrate. A weak model succeeding is a much
stronger indictment than a strong one succeeding.

Measured on the manifest, at native resolution and again on the 224 px image the
model actually sees:

| Fold | kind | AUC (original) | AUC (resized 224) | carried by |
| --- | --- | --- | --- | --- |
| Montgomery | real | 0.647 | 0.672 | entropy → mean |
| Shenzhen | real | 0.853 | 0.775 | Laplacian var → std |
| Montgomery normals + Shenzhen TB | hybrid | **1.000** | **1.000** | width → mean |
| Shenzhen normals + Montgomery TB | hybrid | **1.000** | **1.000** | width → mean |

**The hybrid route is closed.** A manufactured fold is called perfectly by nine
numbers that cannot see anatomy. The obvious objection — that it rides on image
dimensions, which the pipeline resamples away — was tested and does not hold: at
224 px the AUC is still 1.000, now carried by mean brightness. Two sources
differ in exposure and processing in ways that survive any resampling the loader
does. `LIMITATIONS` §3 should stop recommending this construction; a hybrid
cohort is not a clinic and a classifier's accuracy on one says nothing about TB.

**And the real folds are not innocent either.** Shenzhen scores 0.853 at native
resolution, dropping to 0.775 after resampling: inside a genuine two-class fold,
capture statistics alone rank TB above normal well above chance. Montgomery is
milder at 0.647/0.672. This is not a reason to discard the folds — it is far from
1.0, and some of it is real (sicker patients are imaged differently) — but it is
a number that belongs beside every accuracy this project reports, and it is an
argument for adding sites rather than for tuning models. A single-site shortcut
is exactly what a second, third and fourth site would break.

## 3. Running it

```bash
python scripts/audit_clinics.py sources                       # the survey
python scripts/audit_clinics.py calibrate                     # before trusting overlap
python scripts/audit_clinics.py overlap                       # pixel-level leak check
python scripts/audit_clinics.py confound --resize 224         # shortcut check
```

Adding a source means: drop it under `data/raw/`, teach
`tbtrust.data.manifest.infer_clinic_from_path` its provenance rule, add it to
`CLINICS`, rebuild the manifest, then run `calibrate`, `overlap` and `confound`
before it is allowed into a reported rotation.

## 4. What this does not settle

Nothing here has been downloaded. The registry is a survey with its unverified
fields marked, and every count in §1 must be re-derived from the files on
arrival. The audits are built and calibrated against the two sources already in
hand; they are what a new source runs through, not evidence about a source
nobody has fetched yet.
