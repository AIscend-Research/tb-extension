"""Torch Dataset over a manifest. Imports torch/torchvision/PIL, so only pull
this in when you are actually training. The manifest, splits and metrics code
stays torch-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xctb.data.manifest import COHORTS


def cohort_index_map(cohorts=COHORTS) -> dict[str, int]:
    return {c: i for i, c in enumerate(cohorts)}


class CXRDataset:
    """Reads (image, label, cohort_index) rows from a manifest DataFrame.

    Images are loaded as single-channel grayscale. If a path ends in .dcm the
    reader uses pydicom (RSNA ships DICOM); otherwise PIL handles PNG/JPG.
    """

    def __init__(self, manifest: pd.DataFrame, transform=None, cohort_to_idx=None):
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError("CXRDataset needs torch/torchvision installed.") from e

        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform
        self.cohort_to_idx = cohort_to_idx or cohort_index_map()

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_image(self, path: str):
        from PIL import Image

        if str(path).lower().endswith(".dcm"):
            import pydicom

            arr = pydicom.dcmread(path).pixel_array.astype(np.float32)
            arr = 255.0 * (arr - arr.min()) / (np.ptp(arr) + 1e-6)
            return Image.fromarray(arr.astype(np.uint8), mode="L")
        return Image.open(path).convert("L")

    def __getitem__(self, i: int):
        import torch

        row = self.manifest.iloc[i]
        img = self._load_image(row["image_path"])
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        cohort_idx = torch.tensor(self.cohort_to_idx[row["cohort"]], dtype=torch.long)
        return img, label, cohort_idx
