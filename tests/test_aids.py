"""The two lightbox aids: finding them, and the traps they set.

`scripts/measure_fiducial_value.py` measures what a step wedge and a ruler buy.
These tests pin the parts of that path where a wrong answer looks like a right
one -- the detector picking the wrong strip, the base anchor quietly absorbing
bare lightbox, and the black point being freed alongside gamma.
"""

from __future__ import annotations

import numpy as np
import pytest

from tbtrust.physics import aids as A
from tbtrust.physics.density import FilmModel, density_to_transmittance
from tbtrust.physics.fiducials import detect
from tbtrust.physics.film import (
    CaptureParams,
    add_fiducials,
    add_lightbox_aids,
    capture,
    synthetic_chest_density,
)
from tbtrust.physics.invert import invert
from tbtrust.physics.tone import Anchor, fit_tone


@pytest.fixture(scope="module")
def scene():
    """A film with both aids on the lightbox beside it, and its truth."""
    film = FilmModel()
    base, _ = synthetic_chest_density(size=640, rng=np.random.default_rng(3), film=film)
    base, ftruth = add_fiducials(base, film=film)
    px_per_mm = 640 / 432.0
    padded, truth = add_lightbox_aids(base, film=film, px_per_mm=px_per_mm)
    params = CaptureParams(psf_sigma=1.4, tone_gamma=2.6, jpeg_quality=92)
    photo, _ = capture(padded, params, fiducial_truth=ftruth, film=film,
                       rng=np.random.default_rng(7))
    return {"photo": photo, "truth": truth, "px_per_mm": px_per_mm, "film": film,
            "gamma_true": params.tone_gamma}


def test_wedge_is_found_where_it_was_taped(scene):
    v = scene["photo"].astype(float) / 255.0
    readings, _mask, diag = A.detect_wedge(v, detect(v))
    assert readings, diag
    y0, x0, y1, x1 = diag["strip_bbox"]
    ty0, tx0, ty1, tx1 = scene["truth"]["wedge_rect"]
    # Within a few pixels of the real strip, not merely somewhere plausible.
    assert abs(y0 - ty0) < 8 and abs(y1 - ty1) < 8, (diag["strip_bbox"], scene["truth"]["wedge_rect"])
    assert abs(x0 - tx0) < 8 and abs(x1 - tx1) < 8
    # Readings must fall as density rises; that ordering is the whole measurement.
    # Rank correlation rather than strict monotonicity: at the dense end the steps
    # are a couple of 8-bit codes apart, so two neighbours can swap on
    # quantisation alone without the wedge having been misread.
    px = np.array([r["pixel"] for r in sorted(readings, key=lambda r: r["density"])])
    rank = np.argsort(np.argsort(px)).astype(float)
    rho = float(np.corrcoef(np.arange(px.size, dtype=float), rank)[0, 1])
    assert rho < -0.95, (rho, px.tolist())


def test_the_ruler_is_not_mistaken_for_the_wedge(scene):
    """The failure that started this: ranking strips by size picks the ruler.

    A ruler is longer and larger than a wedge, so a size-ranked detector reads 21
    "steps" off it, all at the same pixel value, and hands `fit_tone` a set of
    anchors that constrain gamma to nothing while looking perfectly well-formed.
    """
    v = scene["photo"].astype(float) / 255.0
    _, _, diag = A.detect_wedge(v, detect(v))
    y0, _, y1, _ = diag["strip_bbox"]
    ry0, _, ry1, _ = scene["truth"]["ruler_rect"]
    assert not (y0 >= ry0 - 4 and y1 <= ry1 + 4), "detector locked onto the ruler"
    assert diag["monotonicity"] >= 0.85


def test_ruler_measures_the_scale(scene):
    v = scene["photo"].astype(float) / 255.0
    px_per_mm, _, diag = A.detect_ruler(v, detect(v))
    assert px_per_mm is not None, diag
    rel = abs(px_per_mm - scene["px_per_mm"]) / scene["px_per_mm"]
    assert rel < 0.05, (px_per_mm, scene["px_per_mm"], diag)


def test_ruler_rejects_an_irregular_train():
    """A scale error is silent, so a coincidence must not be accepted as a ruler."""
    rng = np.random.default_rng(0)
    img = np.full((300, 300), 0.8)
    # An elongated strip with dark marks at random positions: elongated, solid,
    # dark enough to be a candidate, and not periodic.
    img[140:170, 20:280] = 0.6
    for x in rng.choice(np.arange(25, 275), size=12, replace=False):
        img[140:170, int(x):int(x) + 3] = 0.2
    px_per_mm, _, diag = A.detect_ruler(img, None)
    assert px_per_mm is None, (px_per_mm, diag)


def test_base_anchor_is_restricted_to_the_sheets_margin(scene):
    """Bare lightbox in frame must not be averaged into base+fog.

    The film's clear margin sits at 0.2 OD and transmits 63%; the lightbox beside
    it transmits everything. `fiducials.detect` takes the base anchor from
    everything outside the field, so a frame wide enough to show the aids is also
    wide enough to bias the one anchor the density scale hangs from.
    """
    v = scene["photo"].astype(float) / 255.0
    fid = detect(v)
    band = A.margin_band(fid, v.shape)
    assert band is not None and band.sum() > 100
    outside = np.asarray(fid.outside_mask, dtype=bool)
    assert float(np.median(v[outside & band])) < float(np.median(v[outside & ~band])), (
        "the margin band should be darker than the bare lightbox beyond it")


def test_aids_selection_is_explicit(scene):
    photo = scene["photo"]
    with pytest.raises(ValueError, match="unknown aids"):
        invert(photo, aids=("wedge", "protractor"))
    # An empty selection still runs: the aids are in frame, neither is used.
    cal = invert(photo, aids=())
    assert cal.tone.method in ("two_point", "one_point")


def test_wedge_fits_gamma_without_freeing_the_black_point(scene):
    """The wedge must buy gamma without reopening the veil degeneracy.

    `c0` and a spatially constant veil are exactly degenerate however many
    anchors there are: a third density identifies gamma, not the split between
    black level and pedestal. Freeing c0 alongside gamma measurably improved the
    fitted gamma and roughly doubled the recovered density error at the same time.
    """
    cal_prior = invert(scene["photo"])
    cal_wedge = invert(scene["photo"], aids=("wedge",))
    assert cal_wedge.tone.method == "three_point"
    assert cal_wedge.tone.c0 == 0.0, "the three-point branch must keep the black point pinned"
    g = scene["gamma_true"]
    assert abs(cal_wedge.tone.gamma - g) <= abs(cal_prior.tone.gamma - g) + 1e-9


def test_fit_tone_recovers_gamma_from_its_own_model():
    """Separates a bad fit from bad anchors; the measurement script leans on it."""
    for g in (1.8, 2.2, 3.0):
        c0, c1 = 0.0, 0.95
        anchors = []
        for d in np.concatenate([[0.2, 3.2], 0.05 + 0.15 * np.arange(21)]):
            lum = float(density_to_transmittance(float(d)))
            v = float(np.clip(round((c0 + c1 * lum ** (1.0 / g)) * 255) / 255, 0, 1))
            anchors.append(Anchor(f"a{d:.2f}", v, 1e-3, float(d), 500))
        t = fit_tone(anchors)
        assert t.method == "three_point"
        assert abs(t.gamma - g) < 0.06, (g, t.gamma)
