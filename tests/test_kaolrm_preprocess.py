from __future__ import annotations

import pytest
import torch


def test_resize_image_to_224_from_larger():
    from nodes.kaolrm_preprocess import _resize_image

    img = torch.rand(1, 512, 512, 3)
    out = _resize_image(img)
    assert out.shape == (1, 224, 224, 3)
    assert out.dtype == torch.float32
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_resize_image_preserves_224_passthrough():
    from nodes.kaolrm_preprocess import _resize_image

    img = torch.rand(2, 224, 224, 3)
    out = _resize_image(img)
    assert out.shape == (2, 224, 224, 3)
    assert torch.allclose(out, img.clamp(0.0, 1.0), atol=1e-6)


def test_resize_image_clamps_out_of_range_values():
    from nodes.kaolrm_preprocess import _resize_image

    img = torch.full((1, 224, 224, 3), 2.5)
    img[..., 0] = -0.3
    out = _resize_image(img)
    assert float(out.max()) <= 1.0
    assert float(out.min()) >= 0.0


def test_resize_image_rejects_bad_shape():
    from nodes.kaolrm_preprocess import _resize_image

    with pytest.raises(ValueError):
        _resize_image(torch.rand(224, 224, 3))
    with pytest.raises(ValueError):
        _resize_image(torch.rand(1, 3, 224, 224))


def test_preprocess_execute_returns_image_and_mask():
    from nodes.kaolrm_preprocess import KaoLRMPreprocess

    img = torch.rand(1, 256, 256, 3)
    out = KaoLRMPreprocess.execute(img, remove_background=False)
    image, mask = out
    assert image.shape == (1, 224, 224, 3)
    assert mask.shape == (1, 224, 224)
    assert float(mask.min()) == 1.0


def test_preprocess_remove_background_invokes_rembg(monkeypatch):
    import numpy as np

    import nodes.kaolrm_preprocess as mod
    from nodes.kaolrm_preprocess import KaoLRMPreprocess

    calls: list[str] = []

    class _StubSession:
        pass

    def fake_get_session(name: str):
        calls.append(name)
        return _StubSession()

    def fake_remove(pil_rgb, session=None, post_process_mask=False):
        from PIL import Image

        w, h = pil_rgb.size
        arr = np.zeros((h, w, 4), dtype=np.uint8)
        arr[..., :3] = 128
        arr[..., 3] = 255
        return Image.fromarray(arr, mode="RGBA")

    monkeypatch.setattr(mod, "_get_rembg_session", fake_get_session)
    monkeypatch.setattr("rembg.remove", fake_remove)

    img = torch.rand(1, 256, 256, 3)
    out = KaoLRMPreprocess.execute(img, remove_background=True, rembg_model="u2net")
    image, mask = out
    assert image.shape == (1, 224, 224, 3)
    assert mask.shape == (1, 224, 224)
    assert calls == ["u2net"]
    assert float(mask.max()) > 0.0
