"""The phantom sheet and the recapture analysis that reads it.

These tests run the analysis against forward-model captures, which is a closed
loop and cannot tell us whether the estimator works on a real phone -- that is
what the pilot in docs/REAL_RECAPTURE.md is for. What they do protect is the
*instrument*: that the sheet is laid out where the analysis thinks it is, that
rectification lands regions on their own rectangles, that the wedge readout
inverts a known transfer, and that the detectability readout responds to contrast
in the direction it must. If any of those breaks, the real captures would be
scored against the wrong rectangles and nothing downstream would say so.

Everything here is at a small canonical size so it stays fast; the pilot runs at
2048.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from tbtrust.physics import phantom as PH
from tbtrust.physics import recapture as RC
from tbtrust.physics.density import density_to_transmittance


@pytest.fixture(scope="module")
def ph():
    return PH.build(size=1024)


def test_layout_is_complete_and_printable(ph):
    s = PH.sanity(ph)
    # A skipped patch is a hole in the grid, and silently reading 24 discs where
    # the ladder promised 30 is the failure this catches.
    assert s["patches_skipped_too_big_for_cell"] == 0
    assert s["n_patches"] == len(PH.DELTA_D_LADDER) * len(PH.SIZE_MM_LADDER)
    assert s["n_steps"] == 11
    assert len(ph.of_kind("wedge")) == PH.WEDGE_STEPS
    # Only the fiducials may sit at film densities; everything printable must be
    # inside what a printer can reach.
    assert s["printable_max_d"] <= ph.d_max_print + 1e-6
    assert s["min_d_is_base_fog"]


def test_regions_do_not_overlap_their_neighbours(ph):
    """Every patch's paired background must be disc-free.

    The background is read as the level the disc's contrast is measured against.
    If the strip overlapped the disc the contrast would be biased toward zero by
    an amount that grows with the contrast itself.
    """
    for patch in ph.of_kind("patch"):
        bg_key = f"bg_d{patch.meta['delta_d']:.3f}_s{patch.meta['size_mm']:g}"
        bg = ph.region(bg_key)
        pv = ph.density[patch.core()]
        bv = ph.density[bg.core()]
        assert np.ptp(bv) < 1e-9, f"{bg_key} is not uniform: the disc leaked into it"
        assert pv.mean() > bv.mean() + 0.5 * patch.meta["delta_d"]


def test_print_image_is_monotone_in_density(ph):
    img = PH.print_image(ph)
    steps = sorted(ph.of_kind("step"), key=lambda r: r.target_d)
    vals = [float(img[r.core()].mean()) for r in steps]
    # Denser target -> more ink -> darker on the sheet, all the way down.
    assert all(a >= b - 1 for a, b in pairwise(vals)), vals


def test_build_from_round_trips(ph):
    same = PH.build_from({"size": 1024})
    assert same.density.shape == ph.density.shape
    assert [r.key for r in same.regions] == [r.key for r in ph.regions]
    with pytest.raises(ValueError):
        PH.build_from({"size": 1024, "not_a_key": 3})


def test_wedge_does_not_fit_is_an_error_not_a_silent_crop():
    # A wedge longer than the sheet must fail loudly: silently cropping it would
    # put unlabelled steps into the calibration.
    with pytest.raises(ValueError, match="does not fit"):
        PH.build(size=1024, wedge_paper_mm=(25.4, 900.0))


def test_rectify_puts_regions_back_on_their_rectangles(ph):
    """A capture with real geometry, rectified, must read its own layout back."""
    photo, _ = RC.simulate_capture(ph, severity=0.1, rng=np.random.default_rng(3))
    rect = RC.rectify(photo, ph)
    assert rect.ok, rect.diagnostics
    readings = RC.read_regions(rect, ph)
    steps = sorted((v for v in readings.values() if v["kind"] == "step"),
                   key=lambda v: v["index"])
    vals = [v["mean"] for v in steps]
    # If the homography were wrong the staircase would not come back as a
    # staircase -- this is the check that the whole readout is addressing the
    # right pixels, and it is the one that fails if the layout ever drifts.
    assert vals == sorted(vals, reverse=True), vals


def test_read_regions_refuses_a_failed_rectification(ph):
    bad = RC.Rectified(np.zeros(ph.density.shape), None, None, None, {"reason": "test"})
    assert RC.read_regions(bad, ph) == {}


def test_wedge_transfer_inverts_a_known_tone_curve(ph):
    """Feed the readout a synthetic capture with a known gamma; it must undo it."""
    d = PH.with_wedge(ph)
    tau = density_to_transmittance(d)
    img = np.clip(tau / tau.max(), 0, 1) ** (1 / 2.2)          # a plausible ISP curve
    rect = RC.Rectified(img, None, np.eye(3), None)
    readings = RC.read_regions(rect, ph)
    to_od, diag = RC.wedge_transfer(readings, ph)
    assert to_od is not None, diag
    assert diag["n_steps_used"] >= 10

    # Steps the wedge did not itself define: the staircase, whose densities we
    # know independently. Recovering those is the actual claim.
    for r in ph.of_kind("step"):
        got = float(to_od(readings[r.key]["mean"]))
        assert abs(got - r.target_d) < 0.06, (r.key, got, r.target_d)


def test_edge_sigma_recovers_known_blur(ph):
    from tbtrust.physics import _ops

    img = density_to_transmittance(PH.with_wedge(ph))
    img = img / img.max()
    got = []
    for sigma in (1.0, 2.0, 4.0):
        rect = RC.Rectified(_ops.gaussian_blur(img, sigma), None, np.eye(3), None)
        got.append(RC._edge_sigma(rect, ph))
    for sigma, g in zip((1.0, 2.0, 4.0), got, strict=True):
        assert abs(g - sigma) < 0.35 * sigma, (sigma, g)


def test_measured_scale_matches_the_layout(ph):
    img = density_to_transmittance(PH.with_wedge(ph))
    rect = RC.Rectified(img / img.max(), None, np.eye(3), None)
    got = RC._measured_px_per_mm(rect, ph)
    # The ruler is read by peak spacing rather than by the dominant Fourier bin,
    # because a tick train's second harmonic can beat its fundamental and the
    # answer then comes out at exactly half -- which is plausible enough to ship.
    assert abs(got - ph.px_per_mm) < 0.1 * ph.px_per_mm, (got, ph.px_per_mm)


def test_characterize_recovers_the_printed_contrasts(ph):
    refs = [RC.simulate_capture(ph, severity=0.0, rng=np.random.default_rng(s))[0]
            for s in (11, 12, 13)]
    truth = RC.characterize(refs, ph)
    assert truth["n_captures_used"] >= 2
    assert truth["reference_reproducibility_od"] < 0.03

    for dd in (0.080, 0.320):
        patch = truth["regions"][f"patch_d{dd:.3f}_s12"]["od"]
        bg = truth["regions"][f"bg_d{dd:.3f}_s12"]["od"]
        assert abs((patch - bg) - dd) < 0.35 * dd, (dd, patch - bg)


def test_detectability_is_monotone_in_contrast(ph):
    """d' must rise with contrast at fixed size. The direction is the claim."""
    refs = [RC.simulate_capture(ph, severity=0.0, rng=np.random.default_rng(s))[0]
            for s in (11, 12)]
    truth = RC.characterize(refs, ph)
    photo, _ = RC.simulate_capture(ph, severity=0.1, rng=np.random.default_rng(21))
    res = RC.score_capture(photo, ph, truth)
    assert res["ok"], res

    by_contrast = {}
    for row in res["detectability"]:
        if "dprime" not in row or row["size_mm"] != 12.0:
            continue
        by_contrast[row["target_delta_d"]] = row["dprime"]
    assert len(by_contrast) >= 4
    lo = by_contrast[min(by_contrast)]
    hi = by_contrast[max(by_contrast)]
    assert hi > lo, by_contrast


def test_confusion_separates_boundary_cases_from_violations():
    rows = [
        # cleared with margin, invisible: a real violation
        {"patch": "a", "dprime": 0.5, "realized_delta_d": 0.10, "predicted_floor": 0.02,
         "predicted_detectable": True, "empirically_detectable": False},
        # a hair over the floor, a hair under d' = k: the bound being about right
        {"patch": "b", "dprime": 4.6, "realized_delta_d": 0.021, "predicted_floor": 0.020,
         "predicted_detectable": True, "empirically_detectable": False},
    ]
    c = RC.confusion(rows, rose_k=5.0)
    assert c["predicted_detectable_but_invisible"] == 2
    assert c["clear_violations"] == 1
    assert c["clear_violation_examples"] == ["a"]
