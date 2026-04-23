"""FLAME UV-layout asset loader.

The FLAME 2020 pkl does not carry `vt` (per-vertex UV coords) or `ft`
(per-face UV indices). These live in `head_template.obj` on the MPI release
page and must be repackaged into a redistributable asset. `FLAMEProjectToUV`
consumes that asset; `FlameCore` stays geometry-only.

License: the UV layout is derived from FLAME 2020 and inherits MPI
non-commercial terms. The asset must be generated locally (see
`scripts/build_flame_uv_template.py`) unless the project clears redistribution
of the layout file explicitly (Outstanding Blocker B1).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
UV_TEMPLATE_PATH = REPO_ROOT / "assets" / "flame_uv_template.npz"
UV_TEMPLATE_URL = "https://flame.is.tue.mpg.de/"

_UV_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def load_uv_template(
    path: str | Path = UV_TEMPLATE_PATH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (vt [V_uv, 2] float32, ft [F, 3] int64).

    - V_uv is typically > 5023 because seam vertices appear multiple times in
      `vt` with different UV positions.
    - F matches `FlameCore.faces` row count (9976 for FLAME 2020 generic).
    - `ft[i]` indexes into `vt`; the paired geometry triangle is `faces[i]`.

    Raises `RuntimeError` naming the path and source URL if the file is absent.
    """
    key = str(Path(path).resolve())
    cached = _UV_CACHE.get(key)
    if cached is not None:
        return cached

    p = Path(path)
    if not p.is_file():
        raise RuntimeError(
            f"FLAME UV template missing at '{p}'. Generate it locally from "
            f"FLAME 2020 'head_template.obj' (available at {UV_TEMPLATE_URL} "
            "after registration) by running "
            "'python scripts/build_flame_uv_template.py /path/to/head_template.obj'."
        )

    data = np.load(p)
    missing = [k for k in ("vt", "ft") if k not in data]
    if missing:
        raise RuntimeError(
            f"FLAME UV template at '{p}' is missing arrays {missing}. "
            f"Expected keys: vt [V_uv, 2] float32, ft [F, 3] int64."
        )

    vt = torch.from_numpy(np.asarray(data["vt"], dtype=np.float32)).contiguous()
    ft = torch.from_numpy(np.asarray(data["ft"], dtype=np.int64)).contiguous()

    if vt.ndim != 2 or vt.shape[1] != 2:
        raise RuntimeError(f"vt must have shape [V_uv, 2], got {tuple(vt.shape)}")
    if ft.ndim != 2 or ft.shape[1] != 3:
        raise RuntimeError(f"ft must have shape [F, 3], got {tuple(ft.shape)}")

    _UV_CACHE[key] = (vt, ft)
    return vt, ft
