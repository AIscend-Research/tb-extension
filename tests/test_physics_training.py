"""Physics as a training input: the parts that would silently measure nothing.

`scripts/measure_physics_in_training.py` compares six arms and the whole result
is a difference of a few points of accuracy. Almost every way this can go wrong
produces a plausible number rather than an error -- a channel that is inert, a
control that is not actually scrambled, a normalisation refitted per image that
erases the between-image signal, a loss weight pointing the wrong way. These pin
those.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tbtrust.data import physics_cache as PC

torch = pytest.importorskip("torch")

from tbtrust.data.dataset import PhysicsCachedDataset  # noqa: E402  (needs torch)

# --- the stem surgery -----------------------------------------------------

def test_the_new_channel_is_inert_at_initialisation():
    """The zero-init claim: the arm starts as the same function as its control.

    If this ever fails, `channel` and `control` no longer start from the same
    place and a difference between them is partly just a reinitialised stem.
    """
    from tbtrust.models.baseline import TBClassifier

    torch.manual_seed(0)
    m = TBClassifier(backbone="resnet18", pretrained=False, in_channels=4).eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        a = m(torch.cat([x, torch.randn(2, 1, 32, 32)], 1))["logit"]
        b = m(torch.cat([x, torch.zeros(2, 1, 32, 32)], 1))["logit"]
        c = m(torch.cat([x, torch.full((2, 1, 32, 32), 7.0)], 1))["logit"]
    assert torch.equal(a, b) and torch.equal(a, c)


def test_surgery_preserves_the_pretrained_rgb_weights():
    from torchvision import models as tvm

    from tbtrust.models.baseline import expand_first_conv

    net = tvm.resnet18(weights=None)
    before = net.conv1.weight.detach().clone()
    expand_first_conv(net, 4)
    assert net.conv1.in_channels == 4
    assert torch.allclose(net.conv1.weight[:, :3], before)
    assert torch.count_nonzero(net.conv1.weight[:, 3:]) == 0


def test_surgery_is_idempotent_and_refuses_an_unexpected_stem():
    from torchvision import models as tvm

    from tbtrust.models.baseline import expand_first_conv

    net = expand_first_conv(tvm.resnet18(weights=None), 4)
    expand_first_conv(net, 4)                      # already 4: no-op, not an error
    assert net.conv1.in_channels == 4
    with pytest.raises(ValueError):
        expand_first_conv(net, 5)                  # 4 -> 5 is not a stem widening


# --- the cache ------------------------------------------------------------

def _fake_item(seed, floor_scale=1.0, abstained=False):
    rng = np.random.default_rng(seed)
    return PC.PhysicsItem(
        photo=rng.integers(0, 255, (16, 16), dtype=np.uint8),
        floor=np.clip(floor_scale * rng.random((16, 16)).astype(np.float32), 1e-4, 1.0),
        mask=np.ones((16, 16), dtype=bool),
        meta={"margin_db": float("nan") if abstained else -3.0 + seed,
              "abstained": abstained, "verdict": "abstain" if abstained else "insufficient"})


def test_cache_roundtrip_survives_the_float16_store(tmp_path):
    it = _fake_item(1)
    PC.save(it, tmp_path / "a.npz")
    back = PC.load(tmp_path / "a.npz")
    assert np.array_equal(back.photo, it.photo)
    assert np.allclose(back.floor, it.floor, atol=1e-3)
    assert back.meta["margin_db"] == it.meta["margin_db"]


def test_normalisation_is_shared_so_images_stay_comparable():
    """Per-image normalisation would erase exactly what the channel carries."""
    dark, bright = _fake_item(2, 0.02), _fake_item(3, 1.0)
    stats = PC.fit_stats([dark, bright])
    a = PC.normalise_floor(dark.floor, stats.log_lo, stats.log_hi)
    b = PC.normalise_floor(bright.floor, stats.log_lo, stats.log_hi)
    assert b.mean() > a.mean() + 0.2          # the worse capture reads as worse
    assert a.min() >= 0.0 and b.max() <= 1.0


def test_an_abstention_is_the_worst_value_not_a_zero():
    """"Could not be measured" must not read as "no degradation"."""
    f = np.full((4, 4), PC.FLOOR_CLIP, dtype=np.float32)
    n = PC.normalise_floor(f, -3.0, 0.0)
    assert np.allclose(n, 1.0)


# --- the dataset arms -----------------------------------------------------

@pytest.fixture
def cache(tmp_path):
    """Six images x two severities of fake cache, plus a manifest over them."""
    rows = []
    for i in range(6):
        for s in (0.0, 1.0):
            path = f"/fake/img{i}.png"
            it = _fake_item(10 * i + int(s), floor_scale=0.05 + 0.15 * i,
                            abstained=(i == 5))
            PC.save(it, tmp_path / f"{PC.cache_key(path, s)}.npz")
        rows.append({"path": f"/fake/img{i}.png", "clinic": "shenzhen",
                     "label": i % 2, "split": "train"})
    df = pd.DataFrame(rows)
    stats = PC.fit_stats([PC.load(tmp_path / f"{PC.cache_key(r['path'], 1.0)}.npz")
                          for r in rows])
    (tmp_path / "stats.json").write_text(json.dumps(stats.to_dict()))
    (tmp_path / "index.json").write_text(json.dumps({"severities": [0.0, 1.0]}))
    return tmp_path, df, stats


def test_control_arm_is_three_channels_and_the_rest_are_four(cache):
    d, df, st = cache
    assert PhysicsCachedDataset(df, d, physics_mode="none").in_channels == 3
    for mode in ("channel", "scramble", "severity"):
        ds = PhysicsCachedDataset(df, d, physics_mode=mode, stats=st)
        assert ds.in_channels == 4
        assert ds[0]["image"].shape[0] == 4


def test_every_arm_sees_the_identical_photograph(cache):
    """The contrast is only attributable if the image is held fixed."""
    d, df, st = cache
    imgs = [PhysicsCachedDataset(df, d, physics_mode=m, stats=st, seed=0)[2]["image"][:3]
            for m in ("none", "channel", "scramble", "severity")]
    for other in imgs[1:]:
        assert torch.equal(imgs[0], other)


def test_scramble_keeps_the_marginal_and_destroys_the_pairing(cache):
    """The control that decides whether a positive result means anything."""
    d, df, st = cache
    real = PhysicsCachedDataset(df, d, physics_mode="channel", stats=st, seed=0)
    fake = PhysicsCachedDataset(df, d, physics_mode="scramble", stats=st, seed=0)
    chans_r = torch.stack([real[i]["image"][3] for i in range(len(df))])
    chans_f = torch.stack([fake[i]["image"][3] for i in range(len(df))])
    # Same pool of maps, different assignment to images.
    assert not torch.allclose(chans_r, chans_f)
    assert abs(float(chans_r.mean()) - float(chans_f.mean())) < 0.25


def test_severity_arm_is_one_constant(cache):
    d, df, st = cache
    ds = PhysicsCachedDataset(df, d, physics_mode="severity", stats=st, seed=0)
    item = ds[0]
    ch = item["image"][3]
    assert float(ch.std()) == 0.0
    assert float(ch.flatten()[0]) == pytest.approx(float(item["severity"]))


def test_loss_weights_point_the_way_they_claim_to(cache):
    d, df, _st = cache
    down = PhysicsCachedDataset(df, d, loss_weight="down", seed=0, epoch_severity=False)
    up = PhysicsCachedDataset(df, d, loss_weight="up", seed=0, epoch_severity=False)
    margins, w_dn, w_up = [], [], []
    for i in range(len(df)):
        margins.append(float(down[i]["margin_db"]))
        w_dn.append(float(down[i]["loss_weight"]))
        w_up.append(float(up[i]["loss_weight"]))
    finite = [i for i, m in enumerate(margins) if np.isfinite(m)]
    best = max(finite, key=lambda i: margins[i])
    worst = min(finite, key=lambda i: margins[i])
    assert w_dn[best] > w_dn[worst]           # down: trust the informative photo
    assert w_up[best] < w_up[worst]           # up: the opposite, by construction
    assert min(w_dn + w_up) >= down.weight_floor - 1e-6


def test_an_abstention_is_weighted_like_the_worst_photograph(cache):
    d, df, _st = cache
    ds = PhysicsCachedDataset(df, d, loss_weight="down", seed=0, epoch_severity=False)
    items = [ds[i] for i in range(len(df))]
    ab = [it for it in items if bool(it["abstained"])]
    assert ab, "fixture should contain an abstention"
    assert float(ab[0]["loss_weight"]) == pytest.approx(ds.weight_floor)


def test_the_dataset_defaults_to_the_grid_the_cache_was_built_on(cache):
    """A cache built on two severities must not be asked for five."""
    d, df, _st = cache
    assert PhysicsCachedDataset(df, d, seed=0).severities == (0.0, 1.0)


def test_a_missing_cache_entry_fails_loudly(cache):
    d, df, _st = cache
    ds = PhysicsCachedDataset(df, d, severities=(0.5,), seed=0, epoch_severity=False)
    with pytest.raises(FileNotFoundError, match="build_physics_cache"):
        ds[0]


def test_unknown_mode_is_rejected_at_construction(cache):
    d, df, _st = cache
    with pytest.raises(ValueError, match="physics_mode"):
        PhysicsCachedDataset(df, d, physics_mode="floor_channel")


# --- the weighted loss ----------------------------------------------------

def test_weighted_loss_reduces_to_the_mean_when_weights_are_equal():
    """The renormalisation must not change the effective learning rate."""
    import torch.nn as nn

    logit = torch.randn(16)
    y = (torch.rand(16) > 0.5).float()
    per = nn.BCEWithLogitsLoss(reduction="none")(logit, y)
    w = torch.full((16,), 0.37)
    weighted = (w * per).sum() / w.sum()
    assert torch.allclose(weighted, nn.BCEWithLogitsLoss()(logit, y))
