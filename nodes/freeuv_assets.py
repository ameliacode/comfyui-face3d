"""Bundled FreeUV reference UV asset loader."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REFERENCE_UV_CACHE: torch.Tensor | None = None

REFERENCE_UV_FILENAME = "freeuv_reference_uv.jpg"
REFERENCE_UV_PATH = Path(__file__).resolve().parents[1] / "assets" / REFERENCE_UV_FILENAME


def load_reference_uv() -> torch.Tensor:
    """Load the bundled reference UV once.

    Returns `[1, H, W, 3]` float32 in `[0, 1]`. Sourced from
    `data-process/resources/uv.jpg` in the upstream FreeUV repo and
    redistributed under CC BY-NC-SA 4.0.
    """
    global _REFERENCE_UV_CACHE
    if _REFERENCE_UV_CACHE is None:
        if not REFERENCE_UV_PATH.exists():
            raise RuntimeError(
                f"Bundled reference UV not found at '{REFERENCE_UV_PATH}'. "
                "This file ships with the repo — check your install. "
                "Upstream source: data-process/resources/uv.jpg at "
                "https://github.com/YangXingchao/FreeUV"
            )
        img = Image.open(REFERENCE_UV_PATH).convert("RGB")
        t = torch.from_numpy(np.array(img)).float() / 255.0
        _REFERENCE_UV_CACHE = t.unsqueeze(0)
    return _REFERENCE_UV_CACHE
