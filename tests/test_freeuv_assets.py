from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def _write_jpeg(path: Path, size: int) -> None:
    arr = np.linspace(0, 255, size * size * 3, dtype=np.uint8).reshape(size, size, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path, format="JPEG")


def test_load_reference_uv_returns_batched_float32(tmp_path, monkeypatch):
    import nodes.freeuv_assets as mod

    asset = tmp_path / "freeuv_reference_uv.jpg"
    _write_jpeg(asset, 64)
    monkeypatch.setattr(mod, "REFERENCE_UV_PATH", asset)
    monkeypatch.setattr(mod, "_REFERENCE_UV_CACHE", None)

    out = mod.load_reference_uv()
    assert out.shape == (1, 64, 64, 3)
    assert out.dtype == torch.float32
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


def test_load_reference_uv_caches_second_call(tmp_path, monkeypatch):
    import nodes.freeuv_assets as mod

    asset = tmp_path / "freeuv_reference_uv.jpg"
    _write_jpeg(asset, 32)
    monkeypatch.setattr(mod, "REFERENCE_UV_PATH", asset)
    monkeypatch.setattr(mod, "_REFERENCE_UV_CACHE", None)

    first = mod.load_reference_uv()
    second = mod.load_reference_uv()
    assert first is second


def test_load_reference_uv_missing_raises(tmp_path, monkeypatch):
    import nodes.freeuv_assets as mod

    missing = tmp_path / "does_not_exist.jpg"
    monkeypatch.setattr(mod, "REFERENCE_UV_PATH", missing)
    monkeypatch.setattr(mod, "_REFERENCE_UV_CACHE", None)

    with pytest.raises(RuntimeError) as exc:
        mod.load_reference_uv()
    assert str(missing) in str(exc.value)
