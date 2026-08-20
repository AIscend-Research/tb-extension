"""Two checks a new clinic has to pass before the rotation may report it.

`data/sources.py` surveys what could be added. This is what runs when something
actually is. Both checks answer questions the split code cannot: `splits.py`
keys on the `clinic` column and trusts it, so a source that is secretly a remix
of an existing one produces a leave-one-clinic-out fold that looks clean and is
not, and a "clinic" assembled by mixing two sources produces a fold whose label
is predictable from the source signature alone.

**1. Overlap.** Perceptual hashing over the pixels, not the filenames. The
Qatar/Dhaka Kaggle bundle re-hosts the NLM images at a different size, under
different names, in a different folder layout -- every metadata-level check
passes and the images are the same images. A difference hash survives the
rescale and the recompression, which is exactly the regime that matters here.

**2. Source confound.** `docs/LIMITATIONS.md` §3 offers a way to manufacture a
two-class fold out of two single-class sources: mix RSNA normals with NIAID
positives. It works arithmetically and it is a trap, because in that fold
`label == source`, and any feature that separates the two machines separates the
two classes for free. `source_confound` measures how far that goes: it fits a
logistic regression on *nothing but* low-level capture statistics -- brightness,
contrast, resolution, noise -- and reports the AUC it reaches. A high AUC means
the fold can be solved without looking at the lungs, and the fold is not
evidence of anything. This is deliberately a weak model: a weak model succeeding
is a much stronger indictment than a strong one succeeding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# 1. Near-duplicate detection
# --------------------------------------------------------------------------

#: Hash side length. 16 gives a 256-bit hash, and the number is not a taste
#: choice -- see `calibrate_threshold`. At the 8 that is conventional for photo
#: deduplication, chest films are so alike that the whole usable band is 2 bits
#: wide (duplicates <=2, different >=4); at 16 it is 24 bits wide. The first
#: version of this audit used 8 with a threshold of 6, which sits inside the
#: known-different distribution and duly reported half of Montgomery as
#: overlapping Shenzhen.
HASH_SIZE = 16

#: Max hamming distance counted as a duplicate, at HASH_SIZE. Calibrated, not
#: guessed: `scripts/audit_clinics.py calibrate` measures a simulated re-bundle
#: landing within 2 bits of its original while the nearest known-different NLM
#: pair is 26 bits away, so 14 sits in the middle of a 24-bit gap. The midpoint
#: rather than the low end, because a missed duplicate is a silent leak and a
#: false one only costs somebody looking at two filenames.
DUPLICATE_THRESHOLD = 14


def dhash(image: np.ndarray, size: int = HASH_SIZE) -> int:
    """Difference hash: which way brightness steps, over a size x size grid.

    Chosen over an exact checksum because the duplicates that matter are not
    byte-identical -- they have been rescaled to 512 px and re-encoded as JPEG on
    the way into a Kaggle bundle. A dhash is invariant to both, and to the
    global brightness and contrast changes that come with a different export
    pipeline, because it stores only the sign of each horizontal gradient.
    """
    from PIL import Image

    img = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("L")
    img = img.resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).ravel()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def hash_image_file(path: str, size: int = HASH_SIZE) -> int:
    from PIL import Image

    return dhash(np.asarray(Image.open(path).convert("L")), size=size)


def simulate_rebundle(path: str, size: int = 512, quality: int = 90) -> np.ndarray:
    """What a source looks like after somebody re-hosts it.

    The positive control for the calibration below, and it is modelled on the
    specific thing being defended against: the Qatar/Dhaka bundle re-hosts the
    NLM images downscaled to 512 px and re-encoded. If the hash cannot recognise
    an image through that, the audit would clear a fold that is 100% duplicated.
    """
    import io

    from PIL import Image

    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("L"))


def calibrate_threshold(known_same, known_different_a, known_different_b,
                        size: int = HASH_SIZE) -> dict:
    """Measure the two distributions the threshold has to separate.

    A duplicate threshold picked by intuition is worthless here, and picking one
    badly is not a quiet failure: at an 8-bit-grid hash with a threshold of 6 --
    both ordinary defaults from photo deduplication -- this audit reported half
    of Montgomery as overlapping Shenzhen, two sets that share no images at all.
    Chest radiographs are far more alike than photographs, so the null
    distribution sits much closer to zero than the usual advice assumes.

    So the threshold is calibrated against two measured distributions:

    * **positives** -- `known_same` images against their own simulated re-bundle.
      This is the duplicate the audit must catch.
    * **negatives** -- all pairs between two sets known to be disjoint.

    Returns both distributions, the gap between them, and a recommended
    threshold at the midpoint. If the gap is not positive, the hash cannot
    separate the two cases at this size and the audit should not be run: that is
    reported rather than papered over with a threshold that splits the overlap.
    """
    pos = [hamming(hash_image_file(p, size), dhash(simulate_rebundle(p), size))
           for p in known_same]
    ha = [hash_image_file(p, size) for p in known_different_a]
    hb = [hash_image_file(p, size) for p in known_different_b]
    neg = [hamming(x, y) for x in ha for y in hb]
    if not pos or not neg:
        return {"usable": False, "note": "need both known-same and known-different pairs"}

    pos_max, neg_min = int(max(pos)), int(min(neg))
    gap = neg_min - pos_max
    return {
        "size": size, "n_bits": size * size,
        "n_positive_pairs": len(pos), "n_negative_pairs": len(neg),
        "positive_max": pos_max, "positive_median": float(np.median(pos)),
        "negative_min": neg_min, "negative_median": float(np.median(neg)),
        "gap_bits": int(gap),
        "recommended_threshold": int((pos_max + neg_min) // 2) if gap > 1 else None,
        "usable": bool(gap > 1),
        "note": ("separable" if gap > 1 else
                 "the nearest known-different pair is as close as the farthest "
                 "known-duplicate: no threshold works at this hash size"),
    }


@dataclass
class OverlapReport:
    """What the pixel-level audit found between two sets of clinics."""

    pairs: list[tuple[str, str, int]]      # (path_a, path_b, hamming distance)
    n_a: int
    n_b: int
    threshold: int

    @property
    def n_overlapping(self) -> int:
        return len({p[0] for p in self.pairs})

    @property
    def fraction_of_a(self) -> float:
        return self.n_overlapping / self.n_a if self.n_a else float("nan")

    def verdict(self, tolerance: float = 0.01) -> str:
        """`clean` / `suspect` / `overlapping`, on the fraction of A also in B.

        A handful of hits at a permissive threshold is what a hash collision
        looks like on natural images -- chest films are far more alike than
        photographs, and a 64-bit dhash will occasionally tie two different
        normal PA films. So a nonzero count is `suspect` and asks for eyes on the
        pairs, and only a sustained fraction is called `overlapping`. Reporting
        one collision as a leak would get this check switched off, which is worse
        than the collisions.
        """
        if not self.pairs:
            return "clean"
        return "overlapping" if self.fraction_of_a > tolerance else "suspect"


def find_overlap(paths_a, paths_b, threshold: int = DUPLICATE_THRESHOLD,
                 size: int = HASH_SIZE) -> OverlapReport:
    """Near-duplicate pairs between two image sets, by perceptual hash.

    `threshold` is in bits and defaults to the calibrated `DUPLICATE_THRESHOLD`.
    Do not raise it by feel -- run `calibrate_threshold` on the sources actually
    in hand and use what it returns, because the gap between "same image, rescaled"
    and "different chest film" is narrower than intuition suggests and depends on
    the sources.
    """
    ha = [(p, hash_image_file(p, size)) for p in paths_a]
    hb = [(p, hash_image_file(p, size)) for p in paths_b]
    pairs = []
    for pa, x in ha:
        for pb, y in hb:
            d = hamming(x, y)
            if d <= threshold:
                pairs.append((pa, pb, d))
    return OverlapReport(pairs=pairs, n_a=len(ha), n_b=len(hb), threshold=threshold)


# --------------------------------------------------------------------------
# 2. Source confound
# --------------------------------------------------------------------------

CAPTURE_FEATURES = ("mean", "std", "p05", "p95", "entropy", "laplacian_var",
                    "height", "width", "aspect")


def capture_features(path: str, resize: int | None = None) -> dict:
    """Low-level statistics of the capture, carrying no anatomy.

    Deliberately crude: global brightness and contrast, the dynamic range, an
    intensity-histogram entropy, a Laplacian variance standing in for sharpness,
    and the raw pixel dimensions. Nothing here can see a cavity or an infiltrate.
    If a model on these features can call the diagnosis, it is calling the
    machine.

    `resize` is what makes the result actionable rather than merely alarming.
    Run on the originals, a confound can be carried entirely by `width` and
    `height` -- two sources exported at different sizes -- and the training
    pipeline resamples everything to `data.image_size` anyway, so that particular
    give-away never reaches the network. Passing `resize=224` recomputes the same
    statistics on the image the model actually sees, which separates "these
    sources differ" from "these sources differ in a way that survives into
    training". Report both; only the second is a finding about the model.
    """
    from PIL import Image

    img = Image.open(path).convert("L")
    original_h, original_w = img.height, img.width
    if resize:
        img = img.resize((resize, resize), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32)
    h, w = (original_h, original_w) if not resize else a.shape
    hist = np.bincount(a.astype(np.uint8).ravel(), minlength=256).astype(float)
    p = hist / max(hist.sum(), 1.0)
    nz = p[p > 0]
    lap = a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    return {
        "mean": float(a.mean()), "std": float(a.std()),
        "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
        "entropy": float(-(nz * np.log2(nz)).sum()),
        "laplacian_var": float(lap.var()),
        "height": float(h), "width": float(w), "aspect": float(w / max(h, 1)),
    }


def source_confound(features, labels, groups=None, n_splits: int = 5,
                    seed: int = 0) -> dict:
    """Can capture statistics alone predict the label? Cross-validated AUC.

    Cross-validated because the feature count is small and the folds this is run
    on are smaller: an in-sample AUC on 150 images and nine features would be
    near 1.0 for any fold at all, including a legitimate one, and would condemn
    everything. `groups` (a patient or study id, when the source has one) keeps
    the same patient out of train and test.

    Read the number as follows. Around 0.5 the label is not recoverable from the
    capture and the fold is honest on this axis. Above ~0.8 the fold is largely
    solvable without anatomy, and reporting a classifier's accuracy on it says
    little about TB. In between is where judgement is needed, and where the
    per-feature coefficients below are worth reading -- if `width` and `height`
    carry it, the two sources were simply exported at different sizes and
    resampling may fix it; if `laplacian_var` and `entropy` carry it, the
    difference is in the machines and it will not wash out.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from ..eval.metrics import roc_auc

    x = np.asarray([[f[k] for k in CAPTURE_FEATURES] for f in features], dtype=float)
    y = np.asarray(labels).astype(int)
    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "note": "single-class input"}

    n_splits = int(min(n_splits, np.bincount(y).min()))
    if n_splits < 2:
        return {"auc": float("nan"), "note": "too few of the minority class to split"}

    if groups is not None:
        splitter = GroupKFold(n_splits=n_splits).split(x, y, np.asarray(groups))
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed).split(x, y)

    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in splitter:
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, C=1.0))
        model.fit(x[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(x[test_idx])[:, 1]

    full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    full.fit(x, y)
    coefs = full[-1].coef_.ravel()
    return {
        "auc": roc_auc(y, oof),
        "n": len(y),
        "n_positive": int(y.sum()),
        "coefficients": dict(zip(CAPTURE_FEATURES, [float(c) for c in coefs],
                                 strict=True)),
        "top_feature": CAPTURE_FEATURES[int(np.argmax(np.abs(coefs)))],
    }


def confound_verdict(auc: float) -> str:
    if not np.isfinite(auc):
        return "unmeasurable"
    if auc >= 0.8:
        return "confounded"
    if auc >= 0.65:
        return "suspect"
    return "acceptable"
