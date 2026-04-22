from __future__ import annotations

import pytest
import torch
from PIL import Image


class _DummyDetailExtractor:
    def __init__(self):
        self.last_kwargs = None

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        return Image.new("RGB", (512, 512), color=(10, 20, 30))


def _freeuv_descriptor() -> dict:
    return {
        "device": "cpu",
        "dtype": "fp32",
        "sd15_root": "/tmp/sd15",
        "clip_root": "/tmp/clip",
        "aligner_path": "/tmp/aligner.bin",
        "detail_path": "/tmp/detail.bin",
        "freeuv_root": None,
    }


def _install_dummy_pipeline(monkeypatch):
    from nodes import freeuv_generate

    freeuv_generate._FREEUV_CACHE.clear()
    extractor = _DummyDetailExtractor()
    handle = {"pipe": object(), "detail_extractor": extractor}
    monkeypatch.setattr(freeuv_generate, "_load_freeuv_pipeline", lambda model: handle)
    return extractor


def test_generate_returns_512_image_via_fallback_reference(monkeypatch):
    from nodes.freeuv_generate import FreeUVGenerate

    extractor = _install_dummy_pipeline(monkeypatch)

    calls = {"count": 0}

    def _spy():
        calls["count"] += 1
        return torch.zeros(1, 32, 32, 3)

    monkeypatch.setattr("nodes.freeuv_generate.load_reference_uv", _spy)

    flaw = torch.rand(1, 512, 512, 3)
    out = FreeUVGenerate.execute(_freeuv_descriptor(), flaw, seed=7)
    image = out[0]
    assert image.shape == (1, 512, 512, 3)
    assert image.dtype == torch.float32
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
    assert calls["count"] == 1
    # Seed flowed through to the generate call.
    assert extractor.last_kwargs["seed"] == 7
    assert extractor.last_kwargs["guidance_scale"] == 1.4
    assert extractor.last_kwargs["num_inference_steps"] == 30


def test_generate_uses_explicit_reference_without_loading_asset(monkeypatch):
    from nodes.freeuv_generate import FreeUVGenerate

    _install_dummy_pipeline(monkeypatch)

    calls = {"count": 0}

    def _spy():
        calls["count"] += 1
        return torch.zeros(1, 32, 32, 3)

    monkeypatch.setattr("nodes.freeuv_generate.load_reference_uv", _spy)

    flaw = torch.rand(1, 512, 512, 3)
    ref = torch.rand(1, 512, 512, 3)
    out = FreeUVGenerate.execute(_freeuv_descriptor(), flaw, reference_uv=ref, seed=1)
    image = out[0]
    assert image.shape == (1, 512, 512, 3)
    assert calls["count"] == 0


def test_generate_rejects_batch_greater_than_one(monkeypatch):
    from nodes.freeuv_generate import FreeUVGenerate

    _install_dummy_pipeline(monkeypatch)
    monkeypatch.setattr("nodes.freeuv_generate.load_reference_uv",
                        lambda: torch.zeros(1, 32, 32, 3))

    flaw = torch.rand(2, 512, 512, 3)
    with pytest.raises(RuntimeError, match="batch size 1"):
        FreeUVGenerate.execute(_freeuv_descriptor(), flaw)


def test_generate_seed_minus_one_samples_a_seed(monkeypatch):
    from nodes.freeuv_generate import FreeUVGenerate

    extractor = _install_dummy_pipeline(monkeypatch)
    monkeypatch.setattr("nodes.freeuv_generate.load_reference_uv",
                        lambda: torch.zeros(1, 32, 32, 3))

    flaw = torch.rand(1, 512, 512, 3)
    FreeUVGenerate.execute(_freeuv_descriptor(), flaw, seed=-1)
    assert extractor.last_kwargs["seed"] != -1
    assert isinstance(extractor.last_kwargs["seed"], int)


def test_cache_key_order_matches_descriptor(monkeypatch):
    from nodes.freeuv_generate import _cache_key

    key = _cache_key(_freeuv_descriptor())
    assert key == ("cpu", "fp32", "/tmp/sd15", "/tmp/clip",
                   "/tmp/aligner.bin", "/tmp/detail.bin", None)
