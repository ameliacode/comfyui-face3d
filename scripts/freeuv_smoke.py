"""Smoke test for FreeUVGenerate end-to-end.

Feeds upstream's sample flaw UV through LoadFreeUV + FreeUVGenerate and saves
the resulting albedo UV. Run from the repo root:

    PYTHONPATH="$HOME/github/ComfyUI:." \
      $HOME/github/ComfyUI/venv311/bin/python scripts/freeuv_smoke.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
FLAW_UV = Path.home() / "github" / "FreeUV" / "data-process" / "results" / "flaw_uv.jpg"
OUT = REPO / "scripts" / "freeuv_smoke_out.png"


def _load_flaw_uv(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def main() -> int:
    from nodes.freeuv_load import LoadFreeUV
    from nodes.freeuv_generate import FreeUVGenerate

    flaw = _load_flaw_uv(FLAW_UV)
    print(f"flaw_uv loaded: {tuple(flaw.shape)} dtype={flaw.dtype}")

    t0 = time.time()
    loader = LoadFreeUV.execute(
        device="cpu", dtype="fp32", i_understand_non_commercial=True
    )
    descriptor = loader.result[0] if hasattr(loader, "result") else loader[0]
    print(f"LoadFreeUV -> {list(descriptor.keys())}  [{time.time()-t0:.1f}s]")

    t0 = time.time()
    result = FreeUVGenerate.execute(
        freeuv_model=descriptor,
        flaw_uv_image=flaw,
        reference_uv=None,
        seed=42,
        guidance_scale=1.4,
        num_inference_steps=30,
    )
    out_tensor = result.result[0] if hasattr(result, "result") else result[0]
    print(f"FreeUVGenerate -> {tuple(out_tensor.shape)} dtype={out_tensor.dtype}  [{time.time()-t0:.1f}s]")

    arr = (out_tensor[0].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(OUT)
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
