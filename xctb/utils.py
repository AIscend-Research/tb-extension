"""Shared helpers: config loading and seeding.

Kept dependency-light on purpose. set_seed touches torch only if torch is
actually installed, so this module imports cleanly in a plain numpy environment.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


def load_config(path: str) -> dict:
    """Load a YAML config, optionally merging a `base:` file it points to.

    A config may set `base: other.yaml` (relative to itself) to inherit defaults
    and override a few keys, so you are not copy-pasting the whole file per run.
    """
    import yaml

    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    base_name = cfg.pop("base", None)
    if base_name:
        base_cfg = load_config(path.parent / base_name)
        base_cfg.update(cfg)
        cfg = base_cfg
    return cfg


def set_seed(seed: int = 0) -> None:
    """Seed python, numpy and (if present) torch for repeatable runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
