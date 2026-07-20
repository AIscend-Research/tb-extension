"""Preprocessing transforms.

Deliberately modest augmentation. The cohorts already differ in contrast and
resolution, and that difference is the domain shift under study, so we resize to
a common size and normalise, then add light augmentation for training. If you
want to erase a specific machine artifact (for example CLAHE to flatten contrast
differences), add it as a per-cohort override and document why in DATA.md,
because it changes what "domain shift" the experiment is measuring.
"""

from __future__ import annotations

# Grayscale CXR normalisation. A single global value is fine to start; refit on
# your training split if you want to be precise.
GRAY_MEAN = [0.5]
GRAY_STD = [0.25]


def build_transforms(image_size: int = 224, train: bool = False):
    """Return a torchvision transform pipeline for single-channel input."""
    from torchvision import transforms

    steps = [transforms.Grayscale(num_output_channels=1)]
    if train:
        steps += [
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
        ]
    else:
        steps += [transforms.Resize((image_size, image_size))]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=GRAY_MEAN, std=GRAY_STD),
    ]
    return transforms.Compose(steps)


def per_cohort_transforms(image_size: int = 224, train: bool = False, overrides: dict | None = None):
    """Map cohort name -> transform, with a shared default and optional overrides.

    overrides example: {"rsna": build_transforms(image_size, train)}  # placeholder
    Use this if a single cohort genuinely needs different handling (a different
    windowing, say). Keep the list of overrides short and justified.
    """
    from xctb.data.manifest import COHORTS

    default = build_transforms(image_size, train)
    table = {c: default for c in COHORTS}
    if overrides:
        table.update(overrides)
    return table
