"""Reproducibility helpers."""

from __future__ import annotations

import os
import random
import zlib

import numpy as np


def seed_everything(seed: int = 0, deterministic: bool = True) -> int:
    """Seed python, numpy, and (if available) torch. Returns the seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


def capture_seed(*parts) -> int:
    """Deterministic seed for one simulated capture, from (path, severity, seed).

    `hash()` on a str is salted per process (PYTHONHASHSEED), so a capture seeded
    from it cannot be regenerated in a later process. That matters here beyond
    ordinary reproducibility: pairing a certificate with a model prediction --
    `eval/physics_deferral.complementarity` is a claim about individual
    photographs -- requires both readings to come off the *same* photo, and with
    a salted seed the second process re-photographs the film differently. CRC32
    over the repr is stable across processes, machines and interpreter runs.
    """
    return zlib.crc32(repr(parts).encode()) % (2**32)
